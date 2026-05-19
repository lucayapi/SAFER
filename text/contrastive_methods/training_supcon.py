"""Entraînement SupCon (SentenceTransformer)."""

from __future__ import annotations

from pathlib import Path

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.data import prepare_text_dataset, split_train_val, train_val_metadata
from contrastive_methods.export import export_st_embeddings
from contrastive_methods.losses.supcon import SupConLoss
from contrastive_methods.metrics import compute_and_save_geometry_metrics
from contrastive_methods.results import TrainingResult
from contrastive_methods.st_common import load_sentence_transformer, train_st_model
from safer_core.io import save_config_resolved
from safer_core.paths import layout_method_output


def run_supcon(cfg: ContrastiveConfig) -> TrainingResult:
    layout = layout_method_output(cfg.method_name, cfg.resolved_output_dir)
    root = Path(layout["root"])
    checkpoints = Path(layout["checkpoints"])
    embeddings_dir = Path(layout["embeddings"])
    metrics_dir = Path(layout["metrics"])

    dataset = prepare_text_dataset(cfg)
    train_idx, val_idx = split_train_val(dataset, cfg)
    train_df, val_df = train_val_metadata(dataset, train_idx, val_idx)

    model = load_sentence_transformer(cfg)
    train_loss = SupConLoss(
        model=model,
        temperature=cfg.supcon_temperature,
        normalize_embeddings=cfg.supcon_normalize_embeddings,
        distance_metric=cfg.distance_metric,
    )

    model, val_geometry, best_score = train_st_model(
        cfg,
        model,
        train_df,
        val_df,
        dataset.text_col,
        train_loss,
        checkpoints,
        train_log_path=metrics_dir / "train_log.csv",
    )

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
        train_log_path=metrics_dir / "train_log.csv",
    )
