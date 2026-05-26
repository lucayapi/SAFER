"""Pipeline TPN : encodeur gelé (modulable) → adaptateur → probas macro → BERTopic."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from macro_transfer.encode import load_target_metadata
from macro_transfer.bertopic_grid import run_macro_bertopic_grid_search
from macro_transfer.intra_bertopic import fit_bertopic_per_macro
from macro_transfer.macro_compression import compute_macro_compression_diagnostics
from macro_transfer.topic_embeddings import resolve_topic_embedding_cfg
from macro_transfer.tpn_adapter import adapt_embeddings_tpn, train_tpn_adapter
from macro_transfer.tpn_encode import (
    default_contrastive_config_path,
    encode_corpus_for_tpn,
    resolve_scgm_encode_log_every_batches,
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
from macro_transfer.report_tables import (
    build_transfer_metrics_comparison,
    embedding_paths_manifest,
)
from macro_transfer.topics_export import (
    build_macro_topic_test_table,
    summarize_topics_by_macro,
)
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


def _merge_bertopic_cfg(
    bertopic_cfg: Dict[str, Any],
    macro_topic_config_path: Optional[str],
    repo_anchor: Path,
) -> Dict[str, Any]:
    cfg = dict(bertopic_cfg or {})
    if not macro_topic_config_path:
        return cfg
    extra_path = resolve_repo_path(macro_topic_config_path, repo_root=repo_anchor)
    with open(extra_path, encoding="utf-8") as f:
        extra = yaml.safe_load(f) or {}
    if "macro_params" in extra:
        cfg["macro_params"] = {**(cfg.get("macro_params") or {}), **extra["macro_params"]}
    for key in ("embedding_space", "grid_search", "diagnostics", "default_params"):
        if key in extra and key not in cfg:
            cfg[key] = extra[key]
    return cfg


def _run_bertopic_phase(
    *,
    out: Path,
    meta_t: pd.DataFrame,
    gating_adapted: pd.DataFrame,
    h_t: np.ndarray,
    h_t_adapted: np.ndarray,
    method_name: str,
    bertopic_cfg: Dict[str, Any],
    topics_export_cfg: Dict[str, Any],
    text_col_t: str,
    repo_anchor: Path,
    corpus_id: Optional[str],
    topic_embedding_mode: Optional[str],
    topic_alpha: Optional[float],
    run_bertopic_grid: bool,
    grid_macros: Optional[Sequence[str]],
    skip_compression_diagnostics: bool,
) -> Dict[str, Any]:
    """Phase BERTopic : compression, grid optionnelle, fit intra-macro."""
    bertopic_cfg = dict(bertopic_cfg)
    if bertopic_cfg.get("enabled", True) is False:
        return {}

    topic_emb_cfg = resolve_topic_embedding_cfg(
        bertopic_cfg,
        cli_mode=topic_embedding_mode,
        cli_alpha=topic_alpha,
    )
    compression_path = None
    grid_path = None
    diagnostics_cfg = dict(bertopic_cfg.get("diagnostics") or {})
    grid_cfg = dict(bertopic_cfg.get("grid_search") or {})

    if (
        not skip_compression_diagnostics
        and diagnostics_cfg.get("enabled", True)
        and diagnostics_cfg.get("compute_compression", True)
    ):
        comp_df = compute_macro_compression_diagnostics(
            h_t,
            h_t_adapted,
            gating_adapted["m_hat"].astype(str).tolist(),
        )
        compression_path = out / "macro_compression_diagnostics.csv"
        comp_df.to_csv(compression_path, index=False)

    do_grid = run_bertopic_grid or bool(grid_cfg.get("enabled", False))
    if do_grid:
        macros_grid = list(grid_macros) if grid_macros else list(grid_cfg.get("macros", ["A0", "A1"]))
        texts_all = meta_t[text_col_t].astype(str).tolist()
        run_macro_bertopic_grid_search(
            texts_all,
            h_t,
            h_t_adapted,
            gating_adapted["m_hat"].astype(str).tolist(),
            macros=macros_grid,
            grid_cfg=grid_cfg,
            output_dir=out,
            bertopic_cfg=bertopic_cfg,
            random_state=int(bertopic_cfg.get("random_state", 42)),
            anchor=repo_anchor if repo_anchor.is_dir() else None,
        )
        grid_path = out / "bertopic_grid_A0_A1.csv"

    top_k_words = int(topics_export_cfg.get("top_k_words", 12))
    top_k_sentences = int(topics_export_cfg.get("top_k_sentences", 5))
    themes_bertopic, assignments_df, bertopic_partial = fit_bertopic_per_macro(
        h_t,
        meta_t,
        gating_adapted,
        method=method_name,
        bertopic_cfg=bertopic_cfg,
        output_dir=out / "topics_bertopic",
        legacy_output_dir=out / "topics_bertopic",
        per_macro_output_root=out / "bertopic",
        run_output_root=out,
        sentence_col=text_col_t,
        top_k_words=top_k_words,
        top_k_sentences=top_k_sentences,
        repo_anchor=repo_anchor if repo_anchor.is_dir() else None,
        corpus_id=corpus_id,
        embeddings_initial=h_t,
        embeddings_adapted=h_t_adapted,
        topic_embedding_cfg=topic_emb_cfg,
    )

    summary_dir = out / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    if not themes_bertopic.empty:
        summarize_topics_by_macro(themes_bertopic).to_csv(
            summary_dir / "topics_summary.csv", index=False
        )
    macro_counts = bertopic_partial.get("macro_topic_counts", {})
    macro_stats = build_macro_topic_test_table(
        macro_counts,
        assignments_df,
        themes_bertopic,
    )
    macro_stats.to_csv(summary_dir / "macro_topic_stats.csv", index=False)

    return {
        "embedding_mode": topic_emb_cfg["mode"],
        "alpha": topic_emb_cfg["alpha"],
        "normalize": topic_emb_cfg["normalize"],
        "macro_topic_counts": macro_counts,
        "macro_topic_stats_path": str(summary_dir / "macro_topic_stats.csv"),
        "warnings": bertopic_partial.get("warnings", []),
        "compression_diagnostics_path": str(compression_path) if compression_path else None,
        "grid_search_path": str(grid_path) if grid_path else None,
    }


def run_tpn_macro_transfer_discovery(
    *,
    checkpoint: str,
    source_data_csv: str,
    target_data_csv: str,
    output_dir: str,
    config: Optional[Dict[str, Any]] = None,
    skip_bertopic: bool = False,
    bertopic_only: bool = False,
    topic_embedding_mode: Optional[str] = None,
    topic_alpha: Optional[float] = None,
    run_bertopic_grid: bool = False,
    grid_macros: Optional[Sequence[str]] = None,
    macro_topic_config_path: Optional[str] = None,
    skip_compression_diagnostics: bool = False,
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
    bertopic_cfg = _merge_bertopic_cfg(
        cfg.get("bertopic") or {},
        macro_topic_config_path,
        repo_anchor,
    )
    topics_export_cfg = cfg.get("topics_export") or {}
    corpus_id = str(cfg.get("corpus") or "").strip() or None

    text_col_s = source_cfg.get("text_col", "sentence")
    label_col_s = source_cfg.get("label_col", "pred_label")
    pred_ok_col_s = source_cfg.get("pred_ok_col", "pred_ok")
    group_col_s = source_cfg.get("group_col", "accident_id")

    text_col_t = target_cfg.get("text_col", cfg.get("text_col", "sentence"))
    label_col_t = target_cfg.get("label_col", cfg.get("label_col", "pred_label"))
    pred_ok_col_t = target_cfg.get("pred_ok_col", cfg.get("pred_ok_col", "pred_ok"))
    group_col_t = target_cfg.get("group_col", group_col_s)

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
    encode_log_every_batches = resolve_scgm_encode_log_every_batches(
        encoding_cfg.get("log_every_batches")
    )

    emb_dir = out / "embeddings"
    transfer_dir = out / "transfer"

    if bertopic_only:
        emb_dir.mkdir(parents=True, exist_ok=True)
        transfer_dir.mkdir(parents=True, exist_ok=True)
        h_t_path = emb_dir / "target_projected.npy"
        h_t_adapted_path = emb_dir / "target_adapted.npy"
        meta_path = transfer_dir / "metadata_with_tpn_macro_probs.csv"
        if not h_t_path.is_file() or not h_t_adapted_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(
                "bertopic_only requiert target_projected.npy, target_adapted.npy et "
                "transfer/metadata_with_tpn_macro_probs.csv dans output_dir"
            )
        h_t = np.load(h_t_path)
        h_t_adapted = np.load(h_t_adapted_path)
        meta_adapted = pd.read_csv(meta_path, low_memory=False)
        meta_t = meta_adapted
        from macro_transfer.constants import MACRO_NAMES

        prob_cols = [f"p_{m}" for m in MACRO_NAMES]
        gating_cols = ["m_hat", "ambiguous", "q_conf", "margin"] + prob_cols
        missing = [c for c in gating_cols if c not in meta_adapted.columns]
        if missing:
            raise ValueError(f"metadata adaptée sans colonnes gating : {missing}")
        gating_adapted = meta_adapted[gating_cols].copy()

        bertopic_summary = {}
        if not skip_bertopic:
            bertopic_summary = _run_bertopic_phase(
                out=out,
                meta_t=meta_t,
                gating_adapted=gating_adapted,
                h_t=h_t,
                h_t_adapted=h_t_adapted,
                method_name=method_name,
                bertopic_cfg=bertopic_cfg,
                topics_export_cfg=topics_export_cfg,
                text_col_t=text_col_t,
                repo_anchor=repo_anchor,
                corpus_id=corpus_id,
                topic_embedding_mode=topic_embedding_mode,
                topic_alpha=topic_alpha,
                run_bertopic_grid=run_bertopic_grid,
                grid_macros=grid_macros,
                skip_compression_diagnostics=skip_compression_diagnostics,
            )
        topic_emb = resolve_topic_embedding_cfg(
            bertopic_cfg, cli_mode=topic_embedding_mode, cli_alpha=topic_alpha
        )
        manifest = {
            "method": method_name,
            "base_encoder": base_method,
            "bertopic_only": True,
            "skip_bertopic": skip_bertopic,
            "topic_embedding_mode": topic_emb["mode"],
            "topic_embedding_alpha": topic_emb["alpha"],
            "topic_embedding_normalize": topic_emb["normalize"],
            "bertopic_summary": bertopic_summary,
            "output_dir": str(out),
        }
        with open(out / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        return manifest

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

    encode_kw_shared = dict(
        contrastive_config=contrastive_path,
        device=str(enc_device),
        batch_size=enc_bs,
        repo_anchor=repo_anchor,
        max_seq_length=max_seq_length,
        scgm_infer_batch_size=scgm_bs,
        log_every_batches=encode_log_every_batches,
    )
    encode_kw_source = {
        **encode_kw_shared,
        "text_col": text_col_s,
        "label_col": label_col_s,
        "pred_ok_col": pred_ok_col_s,
        "group_col": group_col_s,
    }
    encode_kw_target = {
        **encode_kw_shared,
        "text_col": text_col_t,
        "label_col": label_col_t,
        "pred_ok_col": pred_ok_col_t,
        "group_col": group_col_t,
    }

    logger.info(
        "=== Phase 1/5 : encodage %s — source=%d cible=%d (batch=%d device=%s "
        "log_every_batches=%d) ===",
        base_method,
        len(texts_s),
        len(texts_t),
        enc_bs,
        enc_device,
        encode_log_every_batches,
    )
    if base_method == "scgm_text":
        logger.info("--- Encodage SOURCE (BTP) ---")
        h_s = encode_corpus_for_tpn(
            base_method,
            texts_s,
            checkpoint,
            data_csv=source_data_csv,
            emb_csv=scgm_emb_source,
            filter_pred_ok=True,
            log_label="source_btp",
            **encode_kw_source,
        )
        logger.info("--- Encodage CIBLE (test) ---")
        h_t = encode_corpus_for_tpn(
            base_method,
            texts_t,
            checkpoint,
            data_csv=target_data_csv,
            emb_csv=scgm_emb_target,
            filter_pred_ok=False,
            log_label=f"target_{corpus_id or 'test'}",
            **encode_kw_target,
        )
    else:
        logger.info("--- Encodage SOURCE (BTP) ---")
        h_s = encode_corpus_for_tpn(
            base_method,
            texts_s,
            checkpoint,
            filter_pred_ok=True,
            log_label="source_btp",
            **encode_kw_source,
        )
        logger.info("--- Encodage CIBLE (test) ---")
        h_t = encode_corpus_for_tpn(
            base_method,
            texts_t,
            checkpoint,
            filter_pred_ok=False,
            log_label=f"target_{corpus_id or 'test'}",
            **encode_kw_target,
        )
    logger.info(
        "=== Phase 1 terminée : embeddings projetés source=%s cible=%s ===",
        getattr(h_s, "shape", None),
        getattr(h_t, "shape", None),
    )

    if bool(encoding_cfg.get("normalize_embeddings", True)):
        h_s = l2_normalize_np(h_s)
        h_t = l2_normalize_np(h_t)

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
    logger.info("=== Phase 2/5 : gating initial (prototypes projetés) ===")
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
    logger.info(
        "=== Phase 2 terminée : metrics initial acc=%.4f macro_f1=%.4f bal_acc=%.4f ===",
        float(metrics_initial.get("accuracy", float("nan"))),
        float(metrics_initial.get("macro_f1", float("nan"))),
        float(metrics_initial.get("balanced_accuracy", float("nan"))),
    )

    # --- Entraînement adaptateur ---
    train_epochs = int(epochs if epochs is not None else tpn_cfg.get("epochs", 50))
    train_lr = float(learning_rate if learning_rate is not None else tpn_cfg.get("learning_rate", 1e-3))
    train_seed = int(seed if seed is not None else tpn_cfg.get("seed", 42))
    train_device = (
        str(enc_device)
        if str(enc_device).startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    logger.info(
        "=== Phase 3/5 : entraînement adaptateur TPN (%d epochs, lr=%s, device=%s) ===",
        train_epochs,
        train_lr,
        train_device,
    )

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
        log_path=training_dir / "training_log.csv",
    )
    logger.info(
        "=== Phase 3 terminée : loss_final=%.6f (voir %s) ===",
        float(_log_df["loss_total"].iloc[-1]) if len(_log_df) else float("nan"),
        training_dir / "training_log.csv",
    )

    logger.info("=== Phase 4/5 : embeddings adaptés + gating final ===")
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
    logger.info(
        "=== Phase 4 terminée : metrics adapté acc=%.4f macro_f1=%.4f bal_acc=%.4f ===",
        float(metrics_adapted.get("accuracy", float("nan"))),
        float(metrics_adapted.get("macro_f1", float("nan"))),
        float(metrics_adapted.get("balanced_accuracy", float("nan"))),
    )

    bertopic_summary: Dict[str, Any] = {}
    if not skip_bertopic:
        logger.info("=== Phase 5/5 : BERTopic intra-macro ===")
        bertopic_summary = _run_bertopic_phase(
            out=out,
            meta_t=meta_t,
            gating_adapted=gating_adapted,
            h_t=h_t,
            h_t_adapted=h_t_adapted,
            method_name=method_name,
            bertopic_cfg=bertopic_cfg,
            topics_export_cfg=topics_export_cfg,
            text_col_t=text_col_t,
            repo_anchor=repo_anchor,
            corpus_id=corpus_id,
            topic_embedding_mode=topic_embedding_mode,
            topic_alpha=topic_alpha,
            run_bertopic_grid=run_bertopic_grid,
            grid_macros=grid_macros,
            skip_compression_diagnostics=skip_compression_diagnostics,
        )
        logger.info("=== Phase 5 terminée : BERTopic ===")
    elif skip_bertopic:
        logger.info("=== Phase 5 ignorée (skip_bertopic) ===")
    topic_emb = resolve_topic_embedding_cfg(
        bertopic_cfg, cli_mode=topic_embedding_mode, cli_alpha=topic_alpha
    )

    embed_dim = int(h_s.shape[1]) if h_s.ndim == 2 else 0
    emb_paths = embedding_paths_manifest(out)
    summary_dir = out / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    transfer_cmp = build_transfer_metrics_comparison(
        metrics_initial, metrics_adapted, base_method
    )
    transfer_cmp.to_csv(summary_dir / "transfer_metrics_comparison.csv", index=False)

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
        "embedding_paths": emb_paths,
        "metrics_initial": {k: v for k, v in metrics_initial.items() if k != "classification_report"},
        "metrics_adapted": {k: v for k, v in metrics_adapted.items() if k != "classification_report"},
        "transfer_metrics_comparison_path": str(summary_dir / "transfer_metrics_comparison.csv"),
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
        "embedding_paths": emb_paths,
        "metrics_initial": metrics_initial,
        "metrics_adapted": metrics_adapted,
        "skip_bertopic": skip_bertopic,
        "topic_embedding_mode": topic_emb["mode"],
        "topic_embedding_alpha": topic_emb["alpha"],
        "topic_embedding_normalize": topic_emb["normalize"],
        "bertopic_summary": bertopic_summary,
        "transfer_metrics_comparison_path": str(summary_dir / "transfer_metrics_comparison.csv"),
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
