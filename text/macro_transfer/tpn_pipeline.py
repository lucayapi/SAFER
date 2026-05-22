"""Pipeline TPN : encodeur gelé (modulable) → adaptateur → probas macro → BERTopic."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from macro_transfer.encode import load_target_metadata
from macro_transfer.intra_bertopic import fit_bertopic_per_macro
from macro_transfer.tpn_adapter import adapt_embeddings_tpn, train_tpn_adapter
from macro_transfer.tpn_encode import (
    default_contrastive_config_path,
    encode_corpus_for_tpn,
    scgm_checkpoint_input_mode,
    tpn_method_name,
    validate_encoder_name,
)
from macro_transfer.tpn_eval import (
    compute_coverage_by_threshold,
    evaluate_tpn_transfer,
    save_tpn_eval,
)
from macro_transfer.tpn_gating import build_gating_frame, summarize_gating_stats
from macro_transfer.tpn_prototypes import (
    compute_source_prototypes,
    compute_source_target_prototypes,
    compute_target_prototypes_soft,
    l2_normalize_np,
    macro_probs_from_source_prototypes,
    prototype_distance_table,
)
from macro_transfer.topics_export import summarize_topics_by_macro
from safer_core.paths import resolve_repo_path
from scgm_text.dataset_text_embeddings import load_filtered_metadata

logger = logging.getLogger(__name__)

# Rétrocompat : défaut SoftTriple
METHOD_NAME = "tpn_softtriple"
EXPORT_COLS_BASE = ("sentence", "accident_id", "fact_id", "pred_label", "pred_ok", "doc_id")


def _select_export_columns(meta: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in EXPORT_COLS_BASE if c in meta.columns]
    if not cols:
        return meta.iloc[:, :0].copy()
    return meta[cols].copy()


def _build_metadata_export(
    meta: pd.DataFrame,
    gating: pd.DataFrame,
) -> pd.DataFrame:
    base = _select_export_columns(meta)
    out = base.copy()
    if "doc_id" not in out.columns:
        out["doc_id"] = np.arange(len(out))
    for col in gating.columns:
        out[col] = gating[col].values
    return out


def _compute_prototype_bundle(
    h_s: np.ndarray,
    labels_s: np.ndarray,
    h_t: np.ndarray,
    *,
    tpn_cfg: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    tau = float(tpn_cfg.get("tau", 0.3))
    metric = str(tpn_cfg.get("distance_metric", "euclidean"))
    rho = float(tpn_cfg.get("target_weight_st", 1.0))
    assignment_mode = str(tpn_cfg.get("assignment_mode", "soft"))

    mu_s = compute_source_prototypes(h_s, labels_s)
    q = macro_probs_from_source_prototypes(
        h_t, mu_s, tau=tau, metric=metric, assignment_mode=assignment_mode  # type: ignore[arg-type]
    )
    mu_t = compute_target_prototypes_soft(h_t, q)
    mu_st = compute_source_target_prototypes(h_s, labels_s, h_t, q, rho=rho)
    return {"mu_s": mu_s, "mu_t": mu_t, "mu_st": mu_st, "q": q}


def run_tpn_macro_transfer_discovery(
    *,
    checkpoint: str,
    source_data_csv: str,
    target_data_csv: str,
    output_dir: str,
    config: Optional[Dict[str, Any]] = None,
    skip_bertopic: bool = False,
    device: str = "cuda",
    encode_batch_size: int = 8,
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    seed: Optional[int] = None,
    emb_csv: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = dict(config or {})
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    repo_anchor = Path(cfg.get("repo_anchor", Path(__file__).resolve().parents[1]))

    method_cfg = dict(cfg.get("method") or {})
    base_method = validate_encoder_name(
        method_cfg.get("base_method") or method_cfg.get("name") or "softtriple"
    )
    raw_method_name = method_cfg.get("name")
    if raw_method_name and str(raw_method_name).startswith("tpn_"):
        method_name = str(raw_method_name)
    else:
        method_name = tpn_method_name(base_method)

    source_cfg = cfg.get("source") or {}
    target_cfg = cfg.get("target") or {}
    tpn_cfg = dict(cfg.get("tpn") or {})
    adapter_cfg = dict(cfg.get("adapter") or {})
    loss_weights = dict(cfg.get("loss_weights") or {})
    gating_cfg = dict(cfg.get("gating") or {})
    encoding_cfg = dict(cfg.get("encoding") or {})
    bertopic_cfg = cfg.get("bertopic") or {}
    topics_export_cfg = cfg.get("topics_export") or {}

    text_col_s = source_cfg.get("text_col", "sentence")
    label_col_s = source_cfg.get("label_col", "pred_label")
    pred_ok_col_s = source_cfg.get("pred_ok_col", "pred_ok")
    group_col_s = source_cfg.get("group_col", "accident_id")

    text_col_t = target_cfg.get("text_col", cfg.get("text_col", "sentence"))
    label_col_t = target_cfg.get("label_col", cfg.get("label_col", "pred_label"))
    pred_ok_col_t = target_cfg.get("pred_ok_col", cfg.get("pred_ok_col", "pred_ok"))

    contrastive_config = (
        method_cfg.get("contrastive_config")
        or cfg.get("contrastive_config")
        or str(default_contrastive_config_path(base_method, repo_anchor))
    )
    contrastive_path = resolve_repo_path(contrastive_config, repo_root=repo_anchor)

    scgm_emb_target = emb_csv or method_cfg.get("emb_csv") or cfg.get("emb_csv")
    scgm_emb_source = (
        source_cfg.get("emb_csv")
        or method_cfg.get("source_emb_csv")
        or cfg.get("source_emb_csv")
    )
    if scgm_emb_target:
        scgm_emb_target = str(resolve_repo_path(scgm_emb_target, repo_root=repo_anchor))
    if scgm_emb_source:
        scgm_emb_source = str(resolve_repo_path(scgm_emb_source, repo_root=repo_anchor))

    if base_method == "scgm_text":
        scgm_mode = scgm_checkpoint_input_mode(checkpoint)
        if scgm_mode != "text":
            if not scgm_emb_target:
                raise ValueError(
                    "scgm_text requiert method.emb_csv ou emb_csv (registre test) pour la cible"
                )
            if not scgm_emb_source:
                raise ValueError(
                    "scgm_text en mode precomputed_embeddings requiert source.emb_csv explicite "
                    "pour éviter de mélanger les embeddings source (BTP) et cible (test)."
                )

    enc_device = encoding_cfg.get("device", device)
    enc_bs = int(encoding_cfg.get("encode_batch_size", encode_batch_size))
    scgm_bs = int(encoding_cfg.get("scgm_infer_batch_size", cfg.get("infer_batch_size", 512)))
    max_seq_length = int(encoding_cfg.get("max_seq_length", 256))

    # --- Source BTP (labellé, filtré) ---
    meta_s = load_filtered_metadata(
        source_data_csv,
        label_col=label_col_s,
        pred_ok_col=pred_ok_col_s,
        group_col=group_col_s,
        text_col=text_col_s,
    )
    texts_s = meta_s[text_col_s].astype(str).tolist()
    labels_s = meta_s[label_col_s].astype(str).to_numpy()

    # --- Cible (toutes lignes) ---
    meta_t = load_target_metadata(target_data_csv, text_col=text_col_t)
    texts_t = meta_t[text_col_t].astype(str).tolist()

    encode_kw = dict(
        contrastive_config=contrastive_path,
        device=str(enc_device),
        batch_size=enc_bs,
        repo_anchor=repo_anchor,
        text_col=text_col_s,
        label_col=label_col_s,
        pred_ok_col=pred_ok_col_s,
        group_col=group_col_s,
        max_seq_length=max_seq_length,
        scgm_infer_batch_size=scgm_bs,
    )

    logger.info(
        "Encodage %s source (%d) et cible (%d)",
        base_method,
        len(texts_s),
        len(texts_t),
    )

    if base_method == "scgm_text":
        h_s = encode_corpus_for_tpn(
            base_method,
            texts_s,
            checkpoint,
            data_csv=source_data_csv,
            emb_csv=scgm_emb_source,
            filter_pred_ok=True,
            **encode_kw,
        )
        h_t = encode_corpus_for_tpn(
            base_method,
            texts_t,
            checkpoint,
            data_csv=target_data_csv,
            emb_csv=scgm_emb_target,
            filter_pred_ok=False,
            text_col=text_col_t,
            **encode_kw,
        )
    else:
        h_s = encode_corpus_for_tpn(
            base_method, texts_s, checkpoint, filter_pred_ok=True, **encode_kw
        )
        h_t = encode_corpus_for_tpn(
            base_method,
            texts_t,
            checkpoint,
            filter_pred_ok=False,
            text_col=text_col_t,
            **encode_kw,
        )

    if bool(encoding_cfg.get("normalize_embeddings", True)):
        h_s = l2_normalize_np(h_s)
        h_t = l2_normalize_np(h_t)

    emb_dir = out / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_dir / "source_projected.npy", h_s)
    np.save(emb_dir / "target_projected.npy", h_t)

    proto_dir = out / "prototypes"
    proto_dir.mkdir(parents=True, exist_ok=True)
    transfer_dir = out / "transfer"
    transfer_dir.mkdir(parents=True, exist_ok=True)
    training_dir = out / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase initiale (prototypes sur embeddings projetés) ---
    init_bundle = _compute_prototype_bundle(h_s, labels_s, h_t, tpn_cfg=tpn_cfg)
    prob_initial = init_bundle["q"]
    np.savez(
        proto_dir / "prototypes_initial.npz",
        mu_s=init_bundle["mu_s"],
        mu_t=init_bundle["mu_t"],
        mu_st=init_bundle["mu_st"],
        q=init_bundle["q"],
    )
    prototype_distance_table(
        init_bundle["mu_s"], init_bundle["mu_t"], init_bundle["mu_st"]
    ).to_csv(proto_dir / "prototype_distances_initial.csv", index=False)
    np.save(emb_dir / "prob_macro_initial.npy", prob_initial)

    gating_initial = build_gating_frame(
        prob_initial,
        confidence_threshold=float(gating_cfg.get("confidence_threshold", 0.35)),
        margin_threshold=float(gating_cfg.get("margin_threshold", 0.03)),
        ambiguous_rule=gating_cfg.get("ambiguous_rule", "confidence_or_margin"),
    )
    meta_initial = _build_metadata_export(meta_t, gating_initial)
    meta_initial.to_csv(transfer_dir / "metadata_with_initial_macro_probs.csv", index=False)

    metrics_initial = evaluate_tpn_transfer(
        meta_initial, label_col=label_col_t, pred_ok_col=pred_ok_col_t
    )
    save_tpn_eval(metrics_initial, transfer_dir, "initial")
    with open(transfer_dir / "gating_stats_initial.json", "w", encoding="utf-8") as f:
        json.dump(summarize_gating_stats(gating_initial), f, indent=2, ensure_ascii=False)

    # --- Entraînement adaptateur ---
    train_epochs = int(epochs if epochs is not None else tpn_cfg.get("epochs", 50))
    train_lr = float(learning_rate if learning_rate is not None else tpn_cfg.get("learning_rate", 1e-3))
    train_seed = int(seed if seed is not None else tpn_cfg.get("seed", 42))

    resolved_cfg = {
        "method_name": method_name,
        "base_encoder": base_method,
        "checkpoint": checkpoint,
        "contrastive_config": str(contrastive_path),
        "emb_csv_target": scgm_emb_target,
        "emb_csv_source": scgm_emb_source,
        "tpn": tpn_cfg,
        "adapter": adapter_cfg,
        "loss_weights": loss_weights,
        "gating": gating_cfg,
        "epochs": train_epochs,
        "learning_rate": train_lr,
        "seed": train_seed,
    }
    with open(training_dir / "training_config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_cfg, f, allow_unicode=True, sort_keys=False)

    train_device = (
        str(enc_device)
        if str(enc_device).startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )

    adapter, _log_df = train_tpn_adapter(
        h_s,
        h_t,
        labels_s,
        adapter_cfg=adapter_cfg,
        tpn_cfg=tpn_cfg,
        loss_weights=loss_weights,
        epochs=train_epochs,
        learning_rate=train_lr,
        weight_decay=float(tpn_cfg.get("weight_decay", 1e-4)),
        device=train_device,
        seed=train_seed,
        early_stopping_patience=int(tpn_cfg.get("early_stopping_patience", 10)),
        min_delta=float(tpn_cfg.get("min_delta", 1e-5)),
        log_path=training_dir / "training_log.csv",
    )

    adapter = adapter.to(train_device)
    h_s_adapted = adapt_embeddings_tpn(adapter, h_s, device=train_device)
    h_t_adapted = adapt_embeddings_tpn(adapter, h_t, device=train_device)
    np.save(emb_dir / "source_adapted.npy", h_s_adapted)
    np.save(emb_dir / "target_adapted.npy", h_t_adapted)

    # --- Phase finale ---
    final_bundle = _compute_prototype_bundle(h_s_adapted, labels_s, h_t_adapted, tpn_cfg=tpn_cfg)
    prob_adapted = macro_probs_from_source_prototypes(
        h_t_adapted,
        final_bundle["mu_s"],
        tau=float(tpn_cfg.get("tau", 0.3)),
        metric=str(tpn_cfg.get("distance_metric", "euclidean")),  # type: ignore[arg-type]
        assignment_mode=str(tpn_cfg.get("assignment_mode", "soft")),  # type: ignore[arg-type]
    )
    np.save(emb_dir / "prob_macro_adapted.npy", prob_adapted)
    np.savez(
        proto_dir / "prototypes_final.npz",
        mu_s=final_bundle["mu_s"],
        mu_t=final_bundle["mu_t"],
        mu_st=final_bundle["mu_st"],
        q=final_bundle["q"],
    )
    prototype_distance_table(
        final_bundle["mu_s"], final_bundle["mu_t"], final_bundle["mu_st"]
    ).to_csv(proto_dir / "prototype_distances_final.csv", index=False)

    gating_adapted = build_gating_frame(
        prob_adapted,
        confidence_threshold=float(gating_cfg.get("confidence_threshold", 0.35)),
        margin_threshold=float(gating_cfg.get("margin_threshold", 0.03)),
        ambiguous_rule=gating_cfg.get("ambiguous_rule", "confidence_or_margin"),
    )
    meta_adapted = _build_metadata_export(meta_t, gating_adapted)
    meta_adapted.to_csv(transfer_dir / "metadata_with_tpn_macro_probs.csv", index=False)

    metrics_adapted = evaluate_tpn_transfer(
        meta_adapted, label_col=label_col_t, pred_ok_col=pred_ok_col_t
    )
    save_tpn_eval(metrics_adapted, transfer_dir, "adapted")
    with open(transfer_dir / "gating_stats_adapted.json", "w", encoding="utf-8") as f:
        json.dump(summarize_gating_stats(gating_adapted), f, indent=2, ensure_ascii=False)

    coverage_df = compute_coverage_by_threshold(
        meta_adapted, label_col=label_col_t, pred_ok_col=pred_ok_col_t
    )
    if len(coverage_df):
        coverage_df.to_csv(transfer_dir / "coverage_by_threshold.csv", index=False)

    themes_bertopic = pd.DataFrame()
    if not skip_bertopic:
        top_k_words = int(topics_export_cfg.get("top_k_words", 12))
        top_k_sentences = int(topics_export_cfg.get("top_k_sentences", 5))
        corpus_id = str(cfg.get("corpus") or "").strip() or None
        themes_bertopic, _assign = fit_bertopic_per_macro(
            h_t_adapted,
            meta_t,
            gating_adapted,
            method=method_name,
            bertopic_cfg=bertopic_cfg,
            output_dir=out / "topics_bertopic",
            sentence_col=text_col_t,
            top_k_words=top_k_words,
            top_k_sentences=top_k_sentences,
            repo_anchor=repo_anchor if repo_anchor.is_dir() else None,
            corpus_id=corpus_id,
        )

    summary_dir = out / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    if not themes_bertopic.empty:
        summarize_topics_by_macro(themes_bertopic).to_csv(
            summary_dir / "topics_summary.csv", index=False
        )

    embed_dim = int(h_s.shape[1]) if h_s.ndim == 2 else 0
    tpn_summary = {
        "method": method_name,
        "base_encoder": base_method,
        "checkpoint": str(checkpoint),
        "contrastive_config": str(contrastive_path),
        "emb_csv_target": scgm_emb_target,
        "emb_csv_source": scgm_emb_source,
        "embedding_dim": embed_dim,
        "source_data_csv": str(source_data_csv),
        "target_data_csv": str(target_data_csv),
        "n_source": int(len(meta_s)),
        "n_target": int(len(meta_t)),
        "metrics_initial": {k: v for k, v in metrics_initial.items() if k != "classification_report"},
        "metrics_adapted": {k: v for k, v in metrics_adapted.items() if k != "classification_report"},
    }
    with open(summary_dir / "tpn_summary.json", "w", encoding="utf-8") as f:
        json.dump(tpn_summary, f, indent=2, ensure_ascii=False)

    manifest = {
        "method": method_name,
        "base_encoder": base_method,
        "checkpoint": str(checkpoint),
        "source_data_csv": str(source_data_csv),
        "target_data_csv": str(target_data_csv),
        "output_dir": str(out),
        "n_source": int(len(meta_s)),
        "n_target": int(len(meta_t)),
        "metrics_initial": metrics_initial,
        "metrics_adapted": metrics_adapted,
        "skip_bertopic": skip_bertopic,
    }
    with open(out / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **manifest,
                "metrics_initial": {k: v for k, v in metrics_initial.items() if k != "classification_report"},
                "metrics_adapted": {k: v for k, v in metrics_adapted.items() if k != "classification_report"},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return manifest
