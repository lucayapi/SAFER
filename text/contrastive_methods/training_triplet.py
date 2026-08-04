"""Entraînement Batch Hard Triplet (encodeur HF unifié)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.data import prepare_text_dataset, split_train_val, train_val_metadata
from contrastive_methods.encoder_model import build_contrastive_encoder
from contrastive_methods.export import embeddings_to_dataframe
from contrastive_methods.hf_training_common import (
    TextLabelDataset,
    build_optimizer,
    build_train_loader,
    dataloader_kwargs,
    encode_texts,
    forward_embeddings,
    load_contrastive_checkpoint,
    run_training_epoch,
    save_contrastive_checkpoint,
    use_hidden_cache,
)
from contrastive_methods.losses.softtriple import make_collate_fn
from contrastive_methods.losses.triplet_st import build_batch_triplet_embedding_loss
from contrastive_methods.results import TrainingResult
from contrastive_methods.samplers.pk_batch_sampler import (
    build_pk_batch_sampler,
    resolve_balanced_pk_params,
    resolve_validation_pk_params,
)
from contrastive_methods.hf_training_common import get_device, resolve_autocast_dtype
from contrastive_methods.training_log import TRAIN_LOG_COLUMNS, build_train_log_row, print_epoch_line
from contrastive_methods.triplet_diagnostics import TripletDiagnosticLogger
from safer_core.io import save_config_resolved
from safer_core.paths import layout_method_output


@torch.no_grad()
def _run_val_epoch(
    encoder,
    loss_module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_hidden_cache: bool,
    autocast_dtype,
) -> float:
    encoder.eval()
    loss_module.eval()
    total = 0.0
    batches = 0
    for batch in loader:
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype or torch.float32,
            enabled=autocast_dtype is not None and device.type == "cuda",
        ):
            emb = forward_embeddings(encoder, batch, device, use_hidden_cache=use_hidden_cache)
            loss = loss_module(emb, labels)
            if isinstance(loss, tuple):
                loss = loss[0]
        total += float(loss.detach().float().cpu().item())
        batches += 1
    return total / max(1, batches)


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
    val_use_cache = False
    val_pk_sampler = None
    if len(val_df) > 0 and not cfg.final_fit_full_data:
        val_pk = resolve_validation_pk_params(
            val_df["label_id"].tolist(), cfg.batch_size, seed=cfg.seed + 1
        )
        val_pk_sampler = build_pk_batch_sampler(val_df["label_id"].tolist(), val_pk)
        val_loader, val_use_cache = build_train_loader(
            cfg,
            val_df,
            dataset.text_col,
            encoder,
            dev,
            checkpoints / "backbone_cache_val",
            batch_sampler=val_pk_sampler,
        )

    optimizer = build_optimizer(encoder, [loss_module], cfg.learning_rate)
    autocast_dtype = resolve_autocast_dtype(device, cfg.use_amp)
    scaler = (
        torch.cuda.amp.GradScaler()
        if autocast_dtype == torch.float16 and dev.type == "cuda"
        else None
    )
    log_rows: List[dict] = []
    best_train_loss = float("inf")
    best_metric = float("inf")
    best_epoch = 0
    stale_epochs = 0
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
        val_loss = None
        if val_loader is not None:
            val_pk_sampler.set_epoch(epoch)
            val_loss = _run_val_epoch(
                encoder,
                loss_module,
                val_loader,
                dev,
                use_hidden_cache=val_use_cache,
                autocast_dtype=autocast_dtype,
            )
        print_epoch_line("BatchTriplet", epoch_no, cfg.epochs, train_loss, val_loss=val_loss)
        log_rows.append(build_train_log_row(epoch_no, train_loss, val_loss=val_loss))
        best_train_loss = min(best_train_loss, train_loss)
        monitor = val_loss if val_loss is not None else train_loss
        if monitor < best_metric:
            best_metric = monitor
            best_epoch = epoch_no
            stale_epochs = 0
            best_train_loss = train_loss
            save_contrastive_checkpoint(encoder, best_dir)
        elif val_loader is not None:
            stale_epochs += 1
            if stale_epochs >= max(1, cfg.early_stopping_patience):
                break

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
    save_config_resolved(
        {
            **cfg.extra.get("raw", {}),
            "method_name": cfg.method_name,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "best_train_loss": best_train_loss,
            "best_epoch": best_epoch,
            "epochs_ran": len(log_rows),
            "embeddings": str(emb_path),
        },
        root,
    )
    return TrainingResult(
        embeddings_path=emb_path,
        output_root=root,
        best_train_loss=best_train_loss,
        train_wall_time_sec=train_wall_time_sec,
        train_log_path=metrics_dir / "train_log.csv",
        best_epoch=best_epoch,
        epochs_ran=len(log_rows),
    )
