"""Entraînement Batch Hard Triplet (encodeur HF unifié)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.data import prepare_text_dataset, split_train_val, train_val_metadata
from contrastive_methods.encoder_model import build_contrastive_encoder
from contrastive_methods.eval_geometry import selection_score
from contrastive_methods.export import embeddings_to_dataframe
from contrastive_methods.hf_training_common import (
    TextLabelDataset,
    build_eval_loader,
    build_optimizer,
    build_train_loader,
    dataloader_kwargs,
    encode_texts,
    evaluate_val_geometry_from_loader,
    load_contrastive_checkpoint,
    run_training_epoch,
    save_contrastive_checkpoint,
    use_hidden_cache,
)
from contrastive_methods.losses.softtriple import make_collate_fn
from contrastive_methods.losses.triplet_st import build_batch_triplet_embedding_loss
from contrastive_methods.metrics import compute_and_save_geometry_metrics
from contrastive_methods.results import TrainingResult
from contrastive_methods.samplers.pk_batch_sampler import build_pk_batch_sampler, resolve_balanced_pk_params
from contrastive_methods.hf_training_common import get_device, resolve_autocast_dtype
from contrastive_methods.training_log import TRAIN_LOG_COLUMNS, build_train_log_row
from contrastive_methods.triplet_diagnostics import TripletDiagnosticLogger
from safer_core.io import save_config_resolved
from safer_core.paths import layout_method_output


def _print_triplet_epoch(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    *,
    val_geometry: Optional[Dict[str, Any]] = None,
    selection_metric: str = "eta2_macro_balanced_perc",
) -> None:
    parts = [f"[BatchTriplet epoch={epoch}/{total_epochs}]", f"train_loss={train_loss:.4f}"]
    if val_geometry is not None:
        key = selection_metric
        if key in val_geometry:
            parts.append(f"{key}={float(val_geometry[key]):.4f}")
    print(" | ".join(parts), flush=True)


def run_batch_triplet(cfg: ContrastiveConfig) -> TrainingResult:
    layout = layout_method_output(cfg.method_name, cfg.resolved_output_dir)
    root = Path(layout["root"])
    checkpoints = Path(layout["checkpoints"])
    embeddings_dir = Path(layout["embeddings"])
    metrics_dir = Path(layout["metrics"])

    dataset = prepare_text_dataset(cfg)
    train_idx, val_idx = split_train_val(dataset, cfg)
    train_df, val_df = train_val_metadata(dataset, train_idx, val_idx)

    device = get_device()
    dev = torch.device(device)
    encoder = build_contrastive_encoder(cfg).to(dev)

    diagnostic_logger = None
    if cfg.triplet_log_diagnostics:
        diag_path = Path(layout["logs"]) / "batch_triplet_diagnostics.csv"
        diagnostic_logger = TripletDiagnosticLogger(
            diag_path,
            every_steps=cfg.triplet_diagnostics_every_steps,
        )
    loss_module = build_batch_triplet_embedding_loss(cfg, diagnostic_logger=diagnostic_logger).to(dev)

    cache_dir = checkpoints / "backbone_cache"
    triplet_pk = resolve_balanced_pk_params(
        train_df["label_id"].tolist(),
        cfg.batch_size,
        seed=cfg.seed,
    )
    train_ds = TextLabelDataset(train_df, dataset.text_col)
    pk_sampler = build_pk_batch_sampler(train_ds.labels, triplet_pk)
    if use_hidden_cache(cfg):
        train_loader, use_cache = build_train_loader(
            cfg, train_df, dataset.text_col, encoder, dev, cache_dir, batch_sampler=pk_sampler
        )
    else:
        collate = make_collate_fn(encoder.tokenizer, cfg.max_seq_length)
        train_loader = DataLoader(
            train_ds,
            batch_sampler=pk_sampler,
            collate_fn=collate,
            **dataloader_kwargs(device),
        )
        use_cache = False

    val_loader = None
    if len(val_df) > 0 and not cfg.final_fit_full_data:
        val_loader = build_eval_loader(cfg, val_df, dataset.text_col, encoder, dev)

    optimizer = build_optimizer(encoder, [loss_module], cfg.learning_rate)
    autocast_dtype = resolve_autocast_dtype(device)
    scaler = (
        torch.cuda.amp.GradScaler()
        if autocast_dtype == torch.float16 and dev.type == "cuda"
        else None
    )
    log_rows: List[dict] = []
    best_score = float("-inf")
    best_geometry: dict = {}
    best_dir = checkpoints / "best_model"

    t_train = time.perf_counter()
    for epoch in range(cfg.epochs):
        epoch_no = epoch + 1
        if hasattr(pk_sampler, "set_epoch"):
            pk_sampler.set_epoch(epoch)
        train_loss = run_training_epoch(
            encoder,
            loss_module,
            train_loader,
            optimizer,
            dev,
            train=True,
            use_hidden_cache=use_cache,
            autocast_dtype=autocast_dtype,
            scaler=scaler,
            loss_fn=lambda emb, labels: loss_module(emb, labels),
        )
        if val_loader is not None:
            geom = evaluate_val_geometry_from_loader(encoder, val_loader, val_df, cfg, dev)
            score = selection_score(geom, cfg.selection_metric)
            _print_triplet_epoch(
                epoch_no,
                cfg.epochs,
                train_loss,
                val_geometry=geom,
                selection_metric=cfg.selection_metric,
            )
            log_rows.append(build_train_log_row(epoch_no, train_loss, val_geometry=geom))
            if score > best_score:
                best_score = score
                best_geometry = dict(geom)
                save_contrastive_checkpoint(encoder, best_dir)
        else:
            _print_triplet_epoch(epoch_no, cfg.epochs, train_loss)
            log_rows.append(build_train_log_row(epoch_no, train_loss))
            save_contrastive_checkpoint(encoder, best_dir)

    train_wall_time_sec = time.perf_counter() - t_train

    metrics_dir.mkdir(parents=True, exist_ok=True)
    log_df = pd.DataFrame(log_rows)
    for col in TRAIN_LOG_COLUMNS:
        if col not in log_df.columns:
            log_df[col] = None
    log_df[[c for c in TRAIN_LOG_COLUMNS if c in log_df.columns]].to_csv(
        metrics_dir / "train_log.csv", index=False
    )

    encoder = load_contrastive_checkpoint(cfg, best_dir, device)

    emb_path = embeddings_dir / "final_embeddings.csv"
    texts = dataset.metadata_df[dataset.text_col].astype(str).tolist()
    embeddings = encode_texts(encoder, texts, cfg, device)
    frame = embeddings_to_dataframe(dataset.metadata_df["doc_id"].to_numpy(), embeddings)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(emb_path, index=False)
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
        val_geometry=best_geometry,
        best_eta2_macro_balanced_perc=best_score,
        train_wall_time_sec=train_wall_time_sec,
        train_log_path=metrics_dir / "train_log.csv",
    )
