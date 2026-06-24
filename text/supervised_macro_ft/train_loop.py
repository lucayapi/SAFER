"""Boucle d'entraînement CE."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scgm_text.batch_utils import batch_to_device
from scgm_text.metrics import accuracy, balanced_accuracy, macro_f1
from supervised_macro_ft.model import SupervisedMacroModel


def build_class_weights(
    labels: Sequence[int],
    num_classes: int,
    mode: Optional[str],
) -> Optional[torch.Tensor]:
    if not mode or str(mode).lower() in ("none", "null"):
        return None
    y = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.where(counts > 0, counts, 1.0)
    if str(mode).lower() == "balanced":
        weights = len(y) / (num_classes * counts)
        return torch.tensor(weights, dtype=torch.float32)
    raise ValueError(f"class_weight non supporté : {mode!r}")


def build_optimizer(model: SupervisedMacroModel, train_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    lr_backbone = float(train_cfg.get("lr_backbone", 2e-5))
    lr_head = float(train_cfg.get("lr_head", 1e-3))
    wd = float(train_cfg.get("weight_decay", 0.01))
    groups = []
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": wd})
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    if head_params:
        groups.append({"params": head_params, "lr": lr_head, "weight_decay": wd})
    if not groups:
        groups.append({"params": model.parameters(), "lr": lr_head, "weight_decay": wd})
    return torch.optim.AdamW(groups)


def train_one_epoch(
    model: SupervisedMacroModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, batch["label_ids"])
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_loader(
    model: SupervisedMacroModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    total_loss = 0.0
    n_batches = 0
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        batch = batch_to_device(batch, device)
        logits = model(batch)
        loss = criterion(logits, batch["label_ids"])
        total_loss += float(loss.item())
        n_batches += 1
        pred = logits.argmax(dim=-1)
        y_true.extend(batch["label_ids"].cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": accuracy(yt, yp),
        "macro_f1": macro_f1(yt, yp),
        "balanced_accuracy": balanced_accuracy(yt, yp),
    }


def fit_model(
    model: SupervisedMacroModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    *,
    train_cfg: Dict[str, Any],
    device: torch.device,
    class_weight: Optional[torch.Tensor] = None,
) -> Tuple[SupervisedMacroModel, Dict[str, Any]]:
    """Entraîne avec early stopping optionnel sur val macro_f1."""
    epochs = int(train_cfg.get("epochs", 10))
    patience = int(train_cfg.get("early_stopping_patience", 2))
    criterion = nn.CrossEntropyLoss(
        weight=class_weight.to(device) if class_weight is not None else None
    )
    optimizer = build_optimizer(model, train_cfg)
    best_state: Optional[Dict[str, Any]] = None
    best_score = float("-inf")
    best_metrics: Dict[str, Any] = {}
    stale = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        if val_loader is None:
            best_metrics = {"train_loss": train_loss, "epoch": epoch + 1}
            best_state = {
                "backbone": {k: v.cpu().clone() for k, v in model.backbone.encoder.state_dict().items()},
                "classifier": {k: v.cpu().clone() for k, v in model.classifier.state_dict().items()},
            }
            continue
        val_metrics = evaluate_loader(model, val_loader, device)
        score = float(val_metrics.get(str(train_cfg.get("selection_metric", "macro_f1")), val_metrics["macro_f1"]))
        if score > best_score:
            best_score = score
            stale = 0
            best_metrics = {**val_metrics, "train_loss": train_loss, "epoch": epoch + 1}
            best_state = {
                "backbone": {k: v.cpu().clone() for k, v in model.backbone.encoder.state_dict().items()},
                "classifier": {k: v.cpu().clone() for k, v in model.classifier.state_dict().items()},
            }
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.backbone.encoder.load_state_dict(best_state["backbone"])
        model.classifier.load_state_dict(best_state["classifier"])
    return model, best_metrics
