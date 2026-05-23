"""Entraînement Batch Hard Triplet (SentenceTransformer)."""

from __future__ import annotations

import time
from pathlib import Path

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.losses.triplet_st import build_batch_triplet_loss
from contrastive_methods.samplers.pk_batch_sampler import resolve_balanced_pk_params
from contrastive_methods.triplet_diagnostics import TripletDiagnosticLogger
from contrastive_methods.data import prepare_text_dataset, split_train_val, train_val_metadata
from contrastive_methods.eval_geometry import evaluate_st_val_geometry
from contrastive_methods.export import export_st_embeddings
from contrastive_methods.metrics import compute_and_save_geometry_metrics
from contrastive_methods.results import TrainingResult
from contrastive_methods.st_common import (
    load_sentence_transformer,
    resolve_triplet_distance,
    train_st_model,
)
from safer_core.io import save_config_resolved
from safer_core.paths import layout_method_output


def run_batch_triplet(cfg: ContrastiveConfig) -> TrainingResult:
    layout = layout_method_output(cfg.method_name, cfg.resolved_output_dir)
    root = Path(layout["root"])
    checkpoints = Path(layout["checkpoints"])
    embeddings_dir = Path(layout["embeddings"])
    metrics_dir = Path(layout["metrics"])

    dataset = prepare_text_dataset(cfg)
    train_idx, val_idx = split_train_val(dataset, cfg)
    train_df, val_df = train_val_metadata(dataset, train_idx, val_idx)

    model = load_sentence_transformer(cfg)
    diagnostic_logger = None
    if cfg.triplet_log_diagnostics or (
        (cfg.triplet_implementation or "").strip().lower() == "custom_diagnostics"
    ):
        diag_path = Path(layout["logs"]) / "batch_triplet_diagnostics.csv"
        diagnostic_logger = TripletDiagnosticLogger(
            diag_path,
            every_steps=cfg.triplet_diagnostics_every_steps,
        )
    train_loss = build_batch_triplet_loss(cfg, model, diagnostic_logger=diagnostic_logger)
    triplet_pk = resolve_balanced_pk_params(
        train_df["label_id"].tolist(),
        cfg.batch_size,
        seed=cfg.seed,
    )

    t_train = time.perf_counter()
    model, val_geometry, best_score = train_st_model(
        cfg,
        model,
        train_df,
        val_df,
        dataset.text_col,
        train_loss,
        checkpoints,
        train_log_path=metrics_dir / "train_log.csv",
        triplet_pk=triplet_pk,
    )
    train_wall_time_sec = time.perf_counter() - t_train

    emb_path = embeddings_dir / "final_embeddings.csv"
    export_st_embeddings(model, dataset, emb_path, batch_size=cfg.encode_batch_size)
    compute_and_save_geometry_metrics(emb_path, cfg, metrics_dir)
    save_config_resolved(
        {
            **cfg.extra.get("raw", {}),
            "method_name": cfg.method_name,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "best_eta2_macro_balanced_perc": best_score,
            "embeddings": str(emb_path),
        },
        root,
    )
    return TrainingResult(
        embeddings_path=emb_path,
        output_root=root,
        val_geometry=val_geometry,
        best_eta2_macro_balanced_perc=best_score,
        train_wall_time_sec=train_wall_time_sec,
        train_log_path=metrics_dir / "train_log.csv",
    )
