"""Colonnes et helpers pour train_log.csv (toutes méthodes contrastives)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from metrics.geometry import GEOMETRY_METRIC_KEYS

TRAIN_LOG_COLUMNS: List[str] = (
    ["epoch", "train_loss", "val_loss"]
    + [f"val_{key}" for key in GEOMETRY_METRIC_KEYS]
)


class EpochLossAccumulator:
    """Accumule les loss step pour moyenne par epoch (sans logs HF step)."""

    def __init__(self) -> None:
        self._by_epoch: Dict[int, List[float]] = defaultdict(list)

    @staticmethod
    def _epoch_index(epoch: Optional[float]) -> int:
        if epoch is None:
            return 1
        ep = float(epoch)
        base = int(ep)
        if ep > base + 1e-6:
            return base + 1
        return max(1, base)

    def record(self, loss: float, epoch: Optional[float]) -> None:
        self._by_epoch[self._epoch_index(epoch)].append(float(loss))

    def mean_for_epoch(self, epoch: int) -> Optional[float]:
        vals = self._by_epoch.get(int(epoch), [])
        if not vals:
            return None
        return sum(vals) / len(vals)


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


def mean_train_loss_for_epoch(
    log_history: List[Dict[str, Any]],
    epoch: int,
    *,
    loss_accumulator: Optional[EpochLossAccumulator] = None,
) -> Optional[float]:
    """Moyenne des loss step pour l'epoch (accumulateur prioritaire, sinon log_history HF)."""
    if loss_accumulator is not None:
        acc = loss_accumulator.mean_for_epoch(epoch)
        if acc is not None:
            return acc
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
