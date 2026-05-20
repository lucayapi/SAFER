"""Pipeline transfert macro + découverte topics intra-macro."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from macro_transfer.encode import MethodName, encode_target_corpus
from macro_transfer.gating import apply_macro_gating
from macro_transfer.intra_bertopic import fit_bertopic_per_macro
from macro_transfer.intra_gmm import fit_gmm_per_macro
from macro_transfer.openai_topics import enrich_topics_openai
from macro_transfer.scgm_macro import scgm_macro_probs
from macro_transfer.softtriple_macro import macro_probs_from_checkpoint
from macro_transfer.topics_export import comparison_bertopic_vs_gmm
from macro_transfer.transfer_eval import evaluate_transfer_classification, save_transfer_eval


def _default_themes_by_z(method: str, checkpoint: str) -> Optional[Path]:
    if method != "scgm_text":
        return None
    ckpt = Path(checkpoint)
    root = ckpt.parent.parent if ckpt.name.endswith(".pt") else ckpt.parent
    p = root / "topics" / "themes_by_z.csv"
    return p if p.is_file() else None


def run_macro_transfer_discovery(
    *,
    method: MethodName,
    checkpoint: str,
    data_csv: str,
    emb_csv: Optional[str],
    output_dir: str,
    config: Optional[Dict[str, Any]] = None,
    confidence_threshold: float = 0.5,
    macro_temperature: float = 1.0,
    softtriple_gamma: float = 0.1,
    scgm_tau: Optional[float] = None,
    skip_bertopic: bool = False,
    skip_gmm: bool = False,
    skip_openai: bool = True,
    device: str = "cuda",
    batch_size: int = 512,
) -> Dict[str, Any]:
    """Exécute phase 1 (transfert) puis phase 2–3 (topics, OpenAI optionnel)."""
    cfg = dict(config or {})
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    text_col = cfg.get("text_col", "sentence")
    label_col = cfg.get("label_col", "pred_label")
    pred_ok_col = cfg.get("pred_ok_col", "pred_ok")

    z, meta = encode_target_corpus(
        method,
        checkpoint,
        data_csv,
        emb_csv,
        text_col=text_col,
        label_col=label_col,
        pred_ok_col=pred_ok_col,
        batch_size=int(cfg.get("batch_size", batch_size)),
        device=device,
        contrastive_config=Path(cfg["contrastive_config"]) if cfg.get("contrastive_config") else None,
    )
    emb_dir = out / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_dir / "projected.npy", z)

    if method == "scgm_text":
        prob_y, prob_z, prob_y_z = scgm_macro_probs(
            checkpoint, z, tau=scgm_tau, device=device, batch_size=int(cfg.get("infer_batch_size", 4096))
        )
        np.save(emb_dir / "prob_z_x.npy", prob_z)
        np.save(emb_dir / "prob_y_z.npy", prob_y_z)
    else:
        ckpt_dir = Path(checkpoint)
        if ckpt_dir.is_file():
            ckpt_dir = ckpt_dir.parent
        prob_y = macro_probs_from_checkpoint(
            z,
            ckpt_dir,
            gamma=softtriple_gamma,
            temperature=macro_temperature,
            distance_metric=str(cfg.get("distance_metric", "euclidean")),
        )

    gating = apply_macro_gating(prob_y, confidence_threshold=confidence_threshold)
    transfer_dir = out / "transfer"
    transfer_dir.mkdir(parents=True, exist_ok=True)

    export_meta = meta.copy()
    if "doc_id" not in export_meta.columns:
        export_meta["doc_id"] = np.arange(len(export_meta))
    for col in gating.columns:
        export_meta[col] = gating[col].values
    export_meta.to_csv(transfer_dir / "metadata_with_macro_probs.csv", index=False)

    metrics = evaluate_transfer_classification(export_meta, label_col=label_col, pred_ok_col=pred_ok_col)
    save_transfer_eval(metrics, transfer_dir)

    bertopic_cfg = cfg.get("bertopic", {})
    gmm_cfg = cfg.get("gmm", {})
    topics_export_cfg = cfg.get("topics_export", {})
    openai_cfg = cfg.get("openai", {})
    top_k_words = int(topics_export_cfg.get("top_k_words", 12))
    top_k_sentences = int(topics_export_cfg.get("top_k_sentences", 5))

    themes_bertopic = pd.DataFrame()
    themes_gmm = pd.DataFrame()

    if not skip_bertopic:
        themes_bertopic, _ = fit_bertopic_per_macro(
            z,
            meta,
            gating,
            output_dir=out / "topics_bertopic",
            sentence_col=text_col,
            min_topic_size=int(bertopic_cfg.get("min_topic_size", 10)),
            nr_topics=bertopic_cfg.get("nr_topics"),
            random_state=int(bertopic_cfg.get("random_state", 42)),
            top_k_words=top_k_words,
            top_k_sentences=top_k_sentences,
        )

    if not skip_gmm:
        n_comp = gmm_cfg.get("n_components_per_macro")
        themes_gmm, _ = fit_gmm_per_macro(
            z,
            meta,
            gating,
            output_dir=out / "topics_gmm",
            sentence_col=text_col,
            n_components=n_comp,
            k_min=int(gmm_cfg.get("k_min", 2)),
            k_max=int(gmm_cfg.get("k_max", 8)),
            covariance_type=str(gmm_cfg.get("covariance_type", "full")),
            max_iter=int(gmm_cfg.get("max_iter", 200)),
            n_init=int(gmm_cfg.get("n_init", 1)),
            reg_covar=float(gmm_cfg.get("reg_covar", 1.0e-6)),
            random_state=int(gmm_cfg.get("random_state", 42)),
            top_k_words=top_k_words,
            top_k_sentences=top_k_sentences,
        )

    summary_dir = out / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    comparison_bertopic_vs_gmm(themes_bertopic, themes_gmm).to_csv(
        summary_dir / "comparison_bertopic_vs_gmm.csv", index=False
    )

    if not skip_openai:
        oai_dir = out / "openai"
        oai_dir.mkdir(parents=True, exist_ok=True)
        for sub, name in (
            ("topics_bertopic", "themes_bertopic_openai.csv"),
            ("topics_gmm", "themes_gmm_openai.csv"),
        ):
            themes_path = out / sub / "themes_by_macro.csv"
            if themes_path.is_file():
                enrich_topics_openai(
                    themes_path,
                    oai_dir / name,
                    model=str(openai_cfg.get("model", "gpt-4o-mini")),
                    n_example_texts=int(openai_cfg.get("n_example_texts", 5)),
                    skip_on_error=bool(openai_cfg.get("skip_on_error", True)),
                    request_timeout=openai_cfg.get("request_timeout"),
                )

    manifest = {
        "method": method,
        "checkpoint": str(checkpoint),
        "data_csv": str(data_csv),
        "emb_csv": str(emb_csv) if emb_csv else None,
        "output_dir": str(out),
        "n_units": int(len(meta)),
        "transfer_metrics": metrics,
    }
    with open(out / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest
