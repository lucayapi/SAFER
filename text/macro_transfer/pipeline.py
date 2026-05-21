"""Pipeline transfert macro + découverte topics intra-macro."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from macro_transfer.encode import MethodName, encode_target_corpus
from macro_transfer.gating import (
    apply_macro_gating,
    format_gating_stats_message,
    summarize_gating_stats,
)
from macro_transfer.intra_bertopic import fit_bertopic_per_macro
from macro_transfer.scgm_macro import scgm_macro_probs
from macro_transfer.softtriple_macro import macro_probs_from_checkpoint
from macro_transfer.topics_export import summarize_topics_by_macro
from macro_transfer.transfer_eval import evaluate_transfer_classification, save_transfer_eval


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
    device: str = "cuda",
    batch_size: int = 512,
) -> Dict[str, Any]:
    """Exécute phase 1 (transfert) puis phase 2 (BERTopic + representation OpenAI si activée)."""
    cfg = dict(config or {})
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_anchor = cfg.get("repo_anchor")
    repo_anchor = Path(raw_anchor) if raw_anchor else None

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
    topics_export_cfg = cfg.get("topics_export", {})
    top_k_words = int(topics_export_cfg.get("top_k_words", 12))
    top_k_sentences = int(topics_export_cfg.get("top_k_sentences", 5))

    themes_bertopic = pd.DataFrame()

    if not skip_bertopic:
        include_ambiguous = bool(bertopic_cfg.get("include_ambiguous", False))
        gating_stats = summarize_gating_stats(gating)
        with open(transfer_dir / "gating_stats.json", "w", encoding="utf-8") as f:
            json.dump(gating_stats, f, indent=2, ensure_ascii=False)
        if gating_stats["n_non_ambiguous"] == 0 and not include_ambiguous:
            raise RuntimeError(
                "Aucune unité non ambiguë pour BERTopic (q_conf < seuil pour toutes les lignes).\n"
                + format_gating_stats_message(gating_stats, confidence_threshold=confidence_threshold)
                + "\n\nOptions : abaisser confidence_threshold dans macro_transfer.yaml, "
                "ou bertopic.include_ambiguous: true (topics sur toutes les unités m_hat)."
            )
        themes_bertopic, assignments_bertopic = fit_bertopic_per_macro(
            z,
            meta,
            gating,
            output_dir=out / "topics_bertopic",
            sentence_col=text_col,
            bertopic_cfg=bertopic_cfg,
            min_topic_size=int(bertopic_cfg.get("min_topic_size", 10)),
            nr_topics=bertopic_cfg.get("nr_topics"),
            random_state=int(bertopic_cfg.get("random_state", 42)),
            top_k_words=top_k_words,
            top_k_sentences=top_k_sentences,
            repo_anchor=repo_anchor if repo_anchor and repo_anchor.is_dir() else None,
        )
        themes_path = out / "topics_bertopic" / "themes_by_macro.csv"
        if themes_bertopic.empty or not themes_path.is_file():
            raise RuntimeError(
                "BERTopic n'a produit aucun thème (themes_by_macro.csv absent ou vide). "
                f"Assignations : {len(assignments_bertopic)} lignes.\n"
                + format_gating_stats_message(gating_stats, confidence_threshold=confidence_threshold)
                + f"\n  bertopic.include_ambiguous={include_ambiguous}"
            )

    summary_dir = out / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summarize_topics_by_macro(themes_bertopic).to_csv(
        summary_dir / "topics_summary.csv", index=False
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
