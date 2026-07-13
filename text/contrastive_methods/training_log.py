"""Journal d'entraînement contrastif (train_loss / val_loss)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

TRAIN_LOG_COLUMNS: List[str] = ["epoch", "train_loss", "val_loss"]


def build_train_log_row(
    epoch: int,
    train_loss: Optional[float],
    *,
    val_loss: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }


def print_epoch_line(
    tag: str,
    epoch: int,
    total_epochs: int,
    train_loss: Optional[float],
    *,
    val_loss: Optional[float] = None,
) -> None:
    parts = [f"[{tag} epoch={epoch}/{total_epochs}]"]
    if train_loss is not None:
        parts.append(f"train_loss={train_loss:.4f}")
    else:
        parts.append("train_loss=nan")
    if val_loss is not None:
        parts.append(f"val_loss={val_loss:.4f}")
    print(" | ".join(parts), flush=True)


def mean_train_loss_for_epoch(
    log_rows: List[Dict[str, Any]],
    epoch: int,
) -> Optional[float]:
    vals = [
        float(r["train_loss"])
        for r in log_rows
        if r.get("epoch") == epoch and r.get("train_loss") is not None
    ]
    if not vals:
        return None
    return sum(vals) / len(vals)
