"""Transfert test + BERTopic pour supervised_macro_ft."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from macro_transfer.bertopic_config import enrich_run_config_bertopic
from macro_transfer.bertopic_phase import run_bertopic_phase
from macro_transfer.constants import MACRO_NAMES
from macro_transfer.encode import load_target_metadata
from macro_transfer.frozen_source_prototypes import (
    _build_gating_from_predictions,
    evaluate_macro_predictions,
)
from macro_transfer.supervised_baseline import build_predictions_dataframe, export_test_results
from safer_core.io import load_yaml
from safer_core.paths import resolve_repo_path
from safer_core.test_corpus import (
    default_test_corpus_id,
    resolve_test_paths_from_config,
)
from supervised_macro_ft.paths import (
    resolve_supervised_macro_output_dir,
    supervised_macro_ft_output_dir,
    supervised_macro_output_dir,
)
from scgm_text.collate import make_text_collate_fn
from scgm_text.dataset_text_raw import TextRawDataset
from supervised_macro_ft.checkpoint_io import load_checkpoint, read_checkpoint_config
from supervised_macro_ft.geometry_eval import (
    METHOD_LABEL_TEST,
    evaluate_corpus_geometry_with_ipr,
    geometry_keys_from_row,
    save_geometry_metrics_csv,
)

logger = logging.getLogger(__name__)


def _load_tokenizer(backbone_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(backbone_name, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    return tok


@torch.no_grad()
def encode_texts(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    collate_fn = make_text_collate_fn(tokenizer, max_length)
    dummy_labels = [{"text": t, "label": 0, "index": i} for i, t in enumerate(texts)]
    loader = DataLoader(dummy_labels, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    chunks: list[np.ndarray] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        h = model.encode(input_ids, attention_mask)
        chunks.append(h.detach().cpu().numpy())
    if not chunks:
        return np.zeros((0, 1), dtype=np.float64)
    return np.vstack(chunks).astype(np.float64)


@torch.no_grad()
def predict_corpus(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    macros: Sequence[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    collate_fn = make_text_collate_fn(tokenizer, max_length)
    items = [{"text": t, "label": 0, "index": i} for i, t in enumerate(texts)]
    loader = DataLoader(items, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    prob_chunks: list[np.ndarray] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        probs, _ = model.predict_proba(input_ids, attention_mask)
        prob_chunks.append(probs.detach().cpu().numpy())
    probs = np.vstack(prob_chunks)
    top = probs.argmax(axis=1)
    pred_macro = np.array([str(macros[i]) for i in top], dtype=object)
    confidence = probs.max(axis=1)
    sort_p = np.sort(probs, axis=1)
    margin = sort_p[:, -1] - sort_p[:, -2] if probs.shape[1] >= 2 else np.zeros(len(probs))
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)
    return pred_macro, probs, confidence, margin, entropy


def run_supervised_macro_ft_transfer(config_path: str | Path) -> Dict[str, Any]:
    anchor = Path(__file__).resolve().parents[1]
    cfg = enrich_run_config_bertopic(load_yaml(Path(config_path)), anchor=anchor)

    corpus = str(cfg.get("corpus") or default_test_corpus_id())
    target_cfg = dict(cfg.get("target") or {})
    model_cfg = dict(cfg.get("model") or {})
    exp_cfg = dict(cfg.get("exports") or {})
    bertopic_cfg = dict(cfg.get("bertopic") or {})
    topics_export_cfg = dict(cfg.get("topics_export") or {})
    topic_judge_cfg = dict(cfg.get("topic_judge") or {})

    if "run_bertopic" in cfg:
        run_bertopic = bool(cfg.get("run_bertopic"))
    else:
        run_bertopic = bool(bertopic_cfg.get("enabled", True))
    if bool(cfg.get("skip_bertopic", False)):
        run_bertopic = False

    checkpoint = resolve_repo_path(
        str(cfg.get("checkpoint") or cfg.get("model", {}).get("checkpoint_path")),
        repo_root=anchor,
    )
    ckpt_cfg = read_checkpoint_config(checkpoint)
    backbone_name = str(ckpt_cfg.get("backbone_name", model_cfg.get("backbone_name", "Qwen/Qwen3-Embedding-0.6B")))
    max_length = int(model_cfg.get("max_seq_length", ckpt_cfg.get("max_seq_length", 256)))
    batch_size = int(model_cfg.get("encode_batch_size", model_cfg.get("batch_size", 32)))
    device_str = str(model_cfg.get("device", "cuda"))
    device = torch.device("cuda" if device_str.startswith("cuda") and torch.cuda.is_available() else "cpu")

    _spec, target_csv_auto, _ = resolve_test_paths_from_config(
        {"corpus": corpus, "target": {}}, corpus_id=corpus, anchor=anchor
    )
    target_csv = resolve_repo_path(target_cfg.get("dataset_path", str(target_csv_auto)), repo_root=anchor)
    text_col = str(target_cfg.get("text_col", "sentence"))
    label_col = target_cfg.get("label_col", "pred_label")
    group_col = str(target_cfg.get("group_col", "accident_id"))

    test_meta = load_target_metadata(str(target_csv), text_col=text_col)
    texts = test_meta[text_col].astype(str).tolist()
    macros = list(MACRO_NAMES)
    method_slug = str(cfg.get("method_name", "supervised_macro_ft"))
    method_display = str(cfg.get("method_display_name", "Supervised macro FT (CE)"))

    out_dir = resolve_supervised_macro_output_dir(
        method_slug,
        corpus,
        anchor=anchor,
        output_dir=str(cfg["output_dir"]) if cfg.get("output_dir") else None,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    transfer_dir = out_dir / "transfer"
    emb_dir = out_dir / "embeddings"
    transfer_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    model = load_checkpoint(checkpoint, device=str(device))
    tokenizer = _load_tokenizer(backbone_name)

    cache_path = emb_dir / "target_embeddings.npy"
    reuse = bool(model_cfg.get("reuse_cached_embeddings", True))
    z_target = None
    if reuse and cache_path.is_file():
        arr = np.load(cache_path)
        if arr.shape[0] == len(test_meta):
            z_target = arr
            logger.info("Cache target_embeddings réutilisé: %s", cache_path)

    if z_target is None:
        z_target = encode_texts(
            model, tokenizer, texts, max_length=max_length, batch_size=batch_size, device=device
        )

    raw_emb_csv = target_cfg.get("raw_emb_csv")
    if raw_emb_csv:
        raw_emb_csv = str(resolve_repo_path(str(raw_emb_csv), repo_root=anchor))
    else:
        default_test_emb = anchor / "embeddings" / f"Qwen3-Embedding-0.6B_{corpus}.csv"
        raw_emb_csv = str(default_test_emb) if default_test_emb.is_file() else None

    geom_test: Dict[str, Any] = {}
    if label_col and label_col in test_meta.columns:
        try:
            geom_test = evaluate_corpus_geometry_with_ipr(
                z_target,
                test_meta,
                str(label_col),
                method=METHOD_LABEL_TEST,
                raw_emb_csv=raw_emb_csv,
            )
            save_geometry_metrics_csv(geom_test, transfer_dir / "metrics_geometry.csv")
            logger.info("Géométrie test exportée : %s", transfer_dir / "metrics_geometry.csv")
        except Exception as exc:
            logger.warning("Géométrie test ignorée : %s", exc)

    pred_macro, probs, confidence, margin, entropy = predict_corpus(
        model, tokenizer, texts, macros=macros, max_length=max_length, batch_size=batch_size, device=device
    )

    preds = build_predictions_dataframe(
        test_meta,
        pred_macro,
        probs,
        confidence,
        margin,
        entropy,
        macros=macros,
        method_name=method_display,
        text_col=text_col,
        group_col=group_col,
        label_col=str(label_col) if label_col else None,
    )

    metrics_out: Dict[str, Any] = {
        "method": method_display,
        "n_target": int(len(test_meta)),
        "checkpoint": str(checkpoint),
        "run_bertopic": run_bertopic,
        "judge_enable": bool(run_bertopic and topic_judge_cfg.get("enabled", False)),
        "balanced_accuracy": float("nan"),
        "macro_f1": float("nan"),
        "mean_confidence": float(np.mean(confidence)) if len(confidence) else float("nan"),
        "mean_entropy": float(np.mean(entropy)) if len(entropy) else float("nan"),
    }
    if label_col and label_col in test_meta.columns:
        eval_metrics = evaluate_macro_predictions(
            test_meta[label_col].astype(str).to_numpy(),
            pred_macro,
            probs,
            macros,
        )
        cm = np.asarray(eval_metrics.pop("confusion_matrix"))
        cls_rep = eval_metrics.pop("classification_report")
        metrics_out.update(eval_metrics)
        metrics_out["_confusion_matrix"] = cm
        metrics_out["_classification_report"] = cls_rep

    if geom_test:
        for key, val in geometry_keys_from_row(geom_test).items():
            metrics_out[f"geometry_{key}"] = val
        for col in ("IPR_mean", "IPR_A0", "IPR_A1", "IPR_B", "IPR_C"):
            if col in geom_test:
                metrics_out[f"geometry_{col}"] = geom_test[col]

    export_test_results(out_dir, preds, metrics_out, macros=macros)

    if run_bertopic and bool(exp_cfg.get("save_bertopic_inputs", True)):
        bertopic_cols = ["pred_macro", "confidence"] + [f"prob_{m}" for m in macros]
        bertopic_df = preds[
            [c for c in [group_col, "fact_id", "sentence"] if c in preds.columns] + bertopic_cols
        ]
        bertopic_df.to_csv(transfer_dir / "bertopic_input_all.csv", index=False)
        for m in macros:
            bertopic_df[bertopic_df["pred_macro"] == m].to_csv(
                transfer_dir / f"bertopic_input_{m}.csv", index=False
            )

    bertopic_summary: Dict[str, Any] = {}
    if run_bertopic:
        gating = _build_gating_from_predictions(preds, macros)
        meta_t = test_meta.copy()
        meta_t["m_hat"] = preds["pred_macro"].astype(str).to_numpy()
        bertopic_summary = run_bertopic_phase(
            out=out_dir,
            meta_t=meta_t,
            gating_adapted=gating,
            h_t=z_target,
            h_t_adapted=z_target,
            method_name=method_display,
            bertopic_cfg=bertopic_cfg,
            topics_export_cfg=topics_export_cfg,
            text_col_t=text_col,
            repo_anchor=anchor,
            corpus_id=corpus,
            topic_embedding_mode=None,
            topic_alpha=None,
            run_bertopic_grid=False,
            grid_macros=None,
            skip_compression_diagnostics=True,
            topic_judge_cfg=topic_judge_cfg,
        )

    if exp_cfg.get("save_target_embeddings", True):
        np.save(cache_path, z_target)
        test_meta[[c for c in [group_col, "fact_id", text_col] if c in test_meta.columns]].to_csv(
            emb_dir / "target_embeddings_metadata.csv", index=False
        )

    if not run_bertopic:
        logger.info("BERTopic désactivé (run_bertopic=false)")

    return {
        "output_dir": str(out_dir),
        "transfer_dir": str(transfer_dir),
        "metrics_path": str(transfer_dir / "metrics.json"),
        "geometry_metrics_path": str(transfer_dir / "metrics_geometry.csv"),
        "bertopic_summary": bertopic_summary,
    }
