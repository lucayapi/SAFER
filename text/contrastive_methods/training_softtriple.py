"""Entraînement SoftTriple (boucle PyTorch custom)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from contrastive_methods.center_diagnostics import export_softtriple_center_artifacts
from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.data import prepare_text_dataset, split_train_val, train_val_metadata
from contrastive_methods.eval_geometry import evaluate_hf_val_geometry, selection_score
from contrastive_methods.training_log import TRAIN_LOG_COLUMNS, build_train_log_row
from contrastive_methods.export import embeddings_to_dataframe
from contrastive_methods.losses.softtriple import (
    HFTextEncoder,
    SoftTripleLoss,
    encode_texts_with_hf_encoder,
    make_collate_fn,
)
from contrastive_methods.metrics import compute_and_save_geometry_metrics
from contrastive_methods.results import TrainingResult
from contrastive_methods.st_common import get_device
from scgm_text.dataset_text_embeddings import ID2LABEL, LABEL2ID
from safer_core.io import save_config_resolved
from safer_core.paths import layout_method_output


class _SplitDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str) -> None:
        self.texts = df[text_col].astype(str).tolist()
        self.labels = df["label_id"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return {"text": self.texts[idx], "label": self.labels[idx]}


def _print_softtriple_epoch(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    *,
    val_loss: Optional[float] = None,
    val_geometry: Optional[Dict[str, Any]] = None,
    selection_metric: str = "eta2_macro_balanced_perc",
) -> None:
    parts = [
        f"[SoftTriple epoch={epoch}/{total_epochs}]",
        f"train_loss={train_loss:.4f}",
    ]
    if val_loss is not None:
        parts.append(f"val_loss={val_loss:.4f}")
    if val_geometry is not None:
        key = selection_metric
        if key in val_geometry:
            parts.append(f"{key}={float(val_geometry[key]):.4f}")
    print(" | ".join(parts), flush=True)


def _run_epoch(
    encoder: HFTextEncoder,
    loss_module: SoftTripleLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    train: bool,
) -> float:
    encoder.train(mode=train)
    loss_module.train(mode=train)
    total = 0.0
    n_batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        if train:
            optimizer.zero_grad()
        emb = encoder(input_ids, attention_mask)
        loss, _ = loss_module(emb, labels)
        if train:
            loss.backward()
            optimizer.step()
        total += float(loss.detach().cpu().item())
        n_batches += 1
    return total / max(1, n_batches)


def _softtriple_hyperparams_dict(cfg: ContrastiveConfig) -> Dict[str, Any]:
    return {
        "centers_per_class": cfg.centers_per_class,
        "gamma": cfg.softtriple_gamma,
        "lambda": cfg.softtriple_lambda,
        "delta": cfg.softtriple_delta,
        "tau": cfg.softtriple_tau,
        "center_regularization_type": cfg.center_regularization_type,
        "distance_metric": cfg.distance_metric,
        "center_max_similarity": cfg.center_max_similarity,
        "center_min_distance": cfg.center_min_distance,
        "effective_center_distance_threshold": cfg.effective_center_distance_threshold,
        "effective_center_similarity_threshold": cfg.effective_center_similarity_threshold,
    }


def _maybe_export_centers(
    loss_module: SoftTripleLoss,
    cfg: ContrastiveConfig,
    export_dir: Path,
) -> Optional[Path]:
    if not cfg.export_effective_centers or cfg.center_regularization_type == "none":
        return None
    class_names = [ID2LABEL[i] for i in range(len(LABEL2ID))]
    normalize = loss_module.normalize_centers
    return export_softtriple_center_artifacts(
        loss_module.centers.detach().cpu(),
        export_dir,
        class_names=class_names,
        metric=cfg.distance_metric,
        distance_threshold=cfg.effective_center_distance_threshold,
        similarity_threshold=cfg.effective_center_similarity_threshold,
        normalize_centers=normalize,
        hyperparams=_softtriple_hyperparams_dict(cfg),
    )


def _save_softtriple_checkpoint(
    encoder,
    loss_module: SoftTripleLoss,
    cfg: ContrastiveConfig,
    best_dir: Path,
    *,
    export_centers: bool = True,
) -> None:
    best_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.encoder.state_dict(), best_dir / "hf_model.bin")
    encoder.tokenizer.save_pretrained(str(best_dir))
    torch.save(
        {
            "loss_state": loss_module.state_dict(),
            "config": _softtriple_hyperparams_dict(cfg),
        },
        best_dir / "softtriple_state.pt",
    )
    if export_centers:
        _maybe_export_centers(loss_module, cfg, best_dir)


def run_softtriple(cfg: ContrastiveConfig) -> TrainingResult:
    layout = layout_method_output(cfg.method_name, cfg.resolved_output_dir)
    root = Path(layout["root"])
    checkpoints = Path(layout["checkpoints"])
    embeddings_dir = Path(layout["embeddings"])
    metrics_dir = Path(layout["metrics"])
    run_root = Path(cfg.resolved_output_dir)
    if not run_root.is_absolute():
        from safer_core.paths import TEXT_ROOT

        run_root = TEXT_ROOT / run_root

    dataset = prepare_text_dataset(cfg)
    train_idx, val_idx = split_train_val(dataset, cfg)
    train_df, val_df = train_val_metadata(dataset, train_idx, val_idx)

    device = get_device()
    encoder = HFTextEncoder(
        cfg.backbone_name,
        gradient_checkpointing=cfg.gradient_checkpointing,
    ).to(device)
    loss_module = SoftTripleLoss(
        embedding_dim=encoder.embedding_dim,
        num_classes=len(LABEL2ID),
        centers_per_class=cfg.centers_per_class,
        gamma=cfg.softtriple_gamma,
        la=cfg.softtriple_lambda,
        delta=cfg.softtriple_delta,
        tau=cfg.softtriple_tau,
        center_max_similarity=cfg.center_max_similarity,
        center_min_distance=cfg.center_min_distance,
        distance_metric=cfg.distance_metric,
        center_regularization_type=cfg.center_regularization_type,
    ).to(device)

    collate = make_collate_fn(encoder.tokenizer, cfg.max_seq_length)
    train_loader = DataLoader(
        _SplitDataset(train_df, dataset.text_col),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = None
    if len(val_df) > 0 and not cfg.final_fit_full_data:
        val_loader = DataLoader(
            _SplitDataset(val_df, dataset.text_col),
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            collate_fn=collate,
        )

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(loss_module.parameters()),
        lr=cfg.learning_rate,
    )
    dev = torch.device(device)
    log_rows: List[dict] = []
    best_score = float("-inf")
    best_geometry: dict = {}
    best_dir = checkpoints / "best_model"

    t_train = time.perf_counter()
    for epoch in range(cfg.epochs):
        epoch_no = epoch + 1
        train_loss = _run_epoch(
            encoder, loss_module, train_loader, optimizer, dev, train=True
        )
        val_loss = float("nan")
        if val_loader is not None:
            val_loss = _run_epoch(
                encoder, loss_module, val_loader, optimizer, dev, train=False
            )
            geom = evaluate_hf_val_geometry(encoder, val_df, cfg, dataset.text_col, device)
            score = selection_score(geom, cfg.selection_metric)
            _print_softtriple_epoch(
                epoch_no,
                cfg.epochs,
                train_loss,
                val_loss=val_loss,
                val_geometry=geom,
                selection_metric=cfg.selection_metric,
            )
            log_rows.append(
                build_train_log_row(
                    epoch_no,
                    train_loss,
                    val_geometry=geom,
                    val_loss=val_loss,
                )
            )
            if score > best_score:
                best_score = score
                best_geometry = dict(geom)
                _save_softtriple_checkpoint(encoder, loss_module, cfg, best_dir)
        else:
            _print_softtriple_epoch(epoch_no, cfg.epochs, train_loss)
            log_rows.append(build_train_log_row(epoch_no, train_loss))
            _save_softtriple_checkpoint(encoder, loss_module, cfg, best_dir)

    train_wall_time_sec = time.perf_counter() - t_train

    metrics_dir.mkdir(parents=True, exist_ok=True)
    log_df = pd.DataFrame(log_rows)
    for col in TRAIN_LOG_COLUMNS:
        if col not in log_df.columns:
            log_df[col] = None
    log_df[[c for c in TRAIN_LOG_COLUMNS if c in log_df.columns]].to_csv(
        metrics_dir / "train_log.csv", index=False
    )

    ckpt_path = best_dir / "hf_model.bin"
    try:
        state = torch.load(ckpt_path, map_location=dev, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=dev)
    encoder.encoder.load_state_dict(state)
    loss_state_path = best_dir / "softtriple_state.pt"
    if loss_state_path.is_file():
        try:
            ckpt_loss = torch.load(loss_state_path, map_location=dev, weights_only=True)
        except TypeError:
            ckpt_loss = torch.load(loss_state_path, map_location=dev)
        if isinstance(ckpt_loss, dict) and "loss_state" in ckpt_loss:
            loss_module.load_state_dict(ckpt_loss["loss_state"])

    _maybe_export_centers(loss_module, cfg, run_root)

    emb_path = embeddings_dir / "final_embeddings.csv"
    texts = dataset.metadata_df[dataset.text_col].astype(str).tolist()
    embeddings = encode_texts_with_hf_encoder(
        encoder,
        texts,
        batch_size=cfg.encode_batch_size,
        device=device,
        max_length=cfg.max_seq_length,
    )
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
            "center_regularization_type": cfg.center_regularization_type,
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
