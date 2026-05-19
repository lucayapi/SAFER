"""Colonnes et helpers pour train_log.csv (toutes méthodes contrastives)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from metrics.geometry import GEOMETRY_METRIC_KEYS

TRAIN_LOG_COLUMNS: List[str] = (
    ["epoch", "train_loss", "val_loss"]
    + [f"val_{key}" for key in GEOMETRY_METRIC_KEYS]
)


def geometry_row_to_val_columns(row: Dict[str, Any]) -> Dict[str, Any]:
    """Mappe les clés build_geometry_metrics_row vers colonnes val_* du train_log."""
    return {f"val_{key}": row.get(key) for key in GEOMETRY_METRIC_KEYS}


def build_train_log_row(
    epoch: int,
    train_loss: Optional[float],
    *,
    val_geometry: Optional[Dict[str, Any]] = None,
    val_loss: Optional[float] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    if val_geometry:
        row.update(geometry_row_to_val_columns(val_geometry))
    else:
        for col in TRAIN_LOG_COLUMNS:
            if col.startswith("val_") and col != "val_loss":
                row.setdefault(col, None)
    return row


def mean_train_loss_for_epoch(log_history: List[Dict[str, Any]], epoch: int) -> Optional[float]:
    """Moyenne des loss step HF pour l'epoch donné (fractionnaire inclus)."""
    losses: List[float] = []
    for entry in log_history:
        if "loss" not in entry or "eval" in entry:
            continue
        ep = entry.get("epoch")
        if ep is None:
            continue
        if int(ep) == int(epoch) or abs(float(ep) - float(epoch)) < 0.01:
            val = entry.get("loss")
            if val is not None:
                try:
                    losses.append(float(val))
                except (TypeError, ValueError):
                    pass
    if not losses:
        for entry in reversed(log_history):
            if "loss" in entry and "eval" not in entry:
                try:
                    return float(entry["loss"])
                except (TypeError, ValueError):
                    return None
        return None
    return sum(losses) / len(losses)
