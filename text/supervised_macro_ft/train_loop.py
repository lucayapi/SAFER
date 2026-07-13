"""Boucle d'entraînement CE."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scgm_text.batch_utils import batch_to_device
from scgm_text.metrics import accuracy, balanced_accuracy, macro_f1
from supervised_macro_ft.checkpoint_io import _backbone_state_dict, _load_backbone_state_dict
from supervised_macro_ft.model import SupervisedMacroModel
from supervised_macro_ft.run_logging import log_early_stop, log_fit_start

logger = logging.getLogger(__name__)


def _snapshot_model_state(model: SupervisedMacroModel) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "backbone": {k: v.cpu().clone() for k, v in _backbone_state_dict(model).items()},
        "classifier": {k: v.cpu().clone() for k, v in model.classifier.state_dict().items()},
    }
    if model.use_projector:
        state["projector"] = {k: v.cpu().clone() for k, v in model.projector.state_dict().items()}
    return state


def _restore_model_state(model: SupervisedMacroModel, state: Dict[str, Any]) -> None:
    _load_backbone_state_dict(model, state["backbone"])
    model.classifier.load_state_dict(state["classifier"])
    if "projector" in state and model.use_projector:
        model.projector.load_state_dict(state["projector"])


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
    lr_projector = float(train_cfg.get("lr_projector", 1e-3))
    lr_head = float(train_cfg.get("lr_head", 1e-3))
    wd = float(train_cfg.get("weight_decay", 0.01))
    groups = []
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": wd})
    projector_params = [p for p in model.projector.parameters() if p.requires_grad]
    if projector_params:
        groups.append({"params": projector_params, "lr": lr_projector, "weight_decay": wd})
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    if head_params:
        groups.append({"params": head_params, "lr": lr_head, "weight_decay": wd})
    if not groups:
        groups.append({"params": model.parameters(), "lr": lr_head, "weight_decay": wd})
    return torch.optim.AdamW(groups)


def _resolve_use_amp(train_cfg: Dict[str, Any], model: SupervisedMacroModel, device: torch.device) -> bool:
    if "use_amp" in train_cfg:
        return bool(train_cfg.get("use_amp"))
    return bool(model.has_trainable_backbone and device.type == "cuda")


def train_one_epoch(
    model: SupervisedMacroModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    use_amp: bool = False,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    n_batches = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for batch in loader:
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(batch)
            loss = criterion(logits, batch["label_ids"])
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    denom = max(n_batches, 1)
    return {"train_loss": total_loss / denom}


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
    run_label: str = "train",
) -> Tuple[SupervisedMacroModel, Dict[str, Any], List[Dict[str, Any]]]:
    """Entraîne avec early stopping optionnel sur val macro_f1 ; retourne l'historique epoch par epoch."""
    epochs = int(train_cfg.get("epochs", 30))
    patience = int(train_cfg.get("early_stopping_patience", 2))
    selection_metric = str(train_cfg.get("selection_metric", "balanced_accuracy"))
    criterion = nn.CrossEntropyLoss(
        weight=class_weight.to(device) if class_weight is not None else None
    )
    optimizer = build_optimizer(model, train_cfg)
    use_amp = _resolve_use_amp(train_cfg, model, device)
    log_fit_start(
        run_label,
        epochs=epochs,
        selection_metric=selection_metric,
        use_amp=use_amp,
        n_train=len(train_loader.dataset) if hasattr(train_loader, "dataset") else None,
    )
    best_state: Optional[Dict[str, Any]] = None
    best_score = float("-inf")
    best_metrics: Dict[str, Any] = {}
    stale = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_amp=use_amp,
        )
        train_loss = float(train_metrics["train_loss"])
        row: Dict[str, Any] = {"epoch": epoch + 1, **train_metrics}

        if val_loader is None:
            row["is_best"] = True
            history.append(row)
            best_metrics = {**train_metrics, "epoch": epoch + 1}
            best_state = _snapshot_model_state(model)
            logger.info(
                "[macro_ft] [%s] epoch=%d/%d train_loss=%.4f",
                run_label,
                epoch + 1,
                epochs,
                train_loss,
            )
            continue

        val_metrics = evaluate_loader(model, val_loader, device)
        score = float(val_metrics.get(selection_metric, val_metrics["balanced_accuracy"]))
        is_best = score > best_score
        row.update(
            {
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "selection_score": score,
                "is_best": is_best,
            }
        )
        history.append(row)
        if is_best:
            best_score = score
            stale = 0
            best_metrics = {**val_metrics, **train_metrics, "epoch": epoch + 1}
            best_state = _snapshot_model_state(model)
            logger.info(
                "[macro_ft] [%s] epoch=%d/%d train_loss=%.4f val_loss=%.4f %s=%.4f *best*",
                run_label,
                epoch + 1,
                epochs,
                train_loss,
                val_metrics["loss"],
                selection_metric,
                score,
            )
        else:
            stale += 1
            logger.info(
                "[macro_ft] [%s] epoch=%d/%d train_loss=%.4f val_loss=%.4f %s=%.4f (patience %d/%d)",
                run_label,
                epoch + 1,
                epochs,
                train_loss,
                val_metrics["loss"],
                selection_metric,
                score,
                stale,
                patience,
            )
            if stale >= patience:
                log_early_stop(run_label, epoch=epoch + 1, patience=patience)
                break

    if best_state is not None:
        _restore_model_state(model, best_state)
    return model, best_metrics, history
