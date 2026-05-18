"""Entraînement SoftTriple (boucle PyTorch custom)."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.data import prepare_text_dataset, split_train_val, train_val_metadata
from contrastive_methods.eval_geometry import evaluate_hf_val_geometry, selection_score
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
from scgm_text.dataset_text_embeddings import LABEL2ID
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


def _save_softtriple_checkpoint(encoder, loss_module, cfg, best_dir: Path) -> None:
    best_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.encoder.state_dict(), best_dir / "hf_model.bin")
    encoder.tokenizer.save_pretrained(str(best_dir))
    torch.save(
        {
            "loss_state": loss_module.state_dict(),
            "config": {
                "centers_per_class": cfg.centers_per_class,
                "gamma": cfg.softtriple_gamma,
                "lambda": cfg.softtriple_lambda,
                "delta": cfg.softtriple_delta,
                "tau": cfg.softtriple_tau,
            },
        },
        best_dir / "softtriple_state.pt",
    )


def run_softtriple(cfg: ContrastiveConfig) -> TrainingResult:
    layout = layout_method_output(cfg.method_name, cfg.resolved_output_dir)
    root = Path(layout["root"])
    checkpoints = Path(layout["checkpoints"])
    embeddings_dir = Path(layout["embeddings"])
    metrics_dir = Path(layout["metrics"])

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

    for epoch in range(cfg.epochs):
        train_loss = _run_epoch(
            encoder, loss_module, train_loader, optimizer, dev, train=True
        )
        val_loss = float("nan")
        val_delta = float("nan")
        if val_loader is not None:
            val_loss = _run_epoch(
                encoder, loss_module, val_loader, optimizer, dev, train=False
            )
            geom = evaluate_hf_val_geometry(encoder, val_df, cfg, dataset.text_col, device)
            val_delta = float(geom.get("delta_macro_pct", float("nan")))
            score = selection_score(geom, cfg.selection_metric)
            log_rows.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_delta_macro_pct": val_delta,
                    "val_eta2_macro_balanced": geom.get("eta2_macro_balanced"),
                    "val_rankme_global": geom.get("rankme_global"),
                }
            )
            if score > best_score:
                best_score = score
                best_geometry = dict(geom)
                _save_softtriple_checkpoint(encoder, loss_module, cfg, best_dir)
        else:
            log_rows.append({"epoch": epoch + 1, "train_loss": train_loss})
            _save_softtriple_checkpoint(encoder, loss_module, cfg, best_dir)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(metrics_dir / "train_log.csv", index=False)

    ckpt_path = best_dir / "hf_model.bin"
    try:
        state = torch.load(ckpt_path, map_location=dev, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=dev)
    encoder.encoder.load_state_dict(state)
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
            "best_delta_macro_pct": best_score,
            "embeddings": str(emb_path),
        },
        root,
    )
    return TrainingResult(
        embeddings_path=emb_path,
        output_root=root,
        val_geometry=best_geometry,
        best_delta_macro_pct=best_score,
        train_log_path=metrics_dir / "train_log.csv",
    )
