"""Learning-rate schedulers for SCGM-Text (cosine matches official SCGM-G)."""

from __future__ import annotations

import math
from typing import Any, Optional


def adjust_learning_rate_cos(optimizer, lr: float, epoch: int, num_epochs: int, num_cycles: int) -> float:
    """Cosine schedule from official SCGM-G (epoch is 0-based)."""
    epochs_per_cycle = max(1, math.floor(num_epochs / max(1, num_cycles)))
    new_lr = lr * 0.5 * (1.0 + math.cos(math.pi * (epoch % epochs_per_cycle) / epochs_per_cycle))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    return new_lr


def build_scheduler(optimizer, config: Any) -> Optional[str]:
    """Returns scheduler name ('none' or 'cosine'); stepping is explicit."""
    name = str(getattr(config, "scheduler", "none")).strip().lower()
    if name in ("none", ""):
        return None
    if name == "cosine":
        return "cosine"
    raise ValueError(f"Unknown scheduler: {name!r} (expected none or cosine)")


def step_scheduler(
    optimizer,
    config: Any,
    epoch: int,
    total_epochs: int,
) -> float:
    """Update LR at start of epoch; returns current LR."""
    scheduler = build_scheduler(optimizer, config)
    if scheduler is None:
        return float(optimizer.param_groups[0]["lr"])
    if scheduler == "cosine":
        base_lr = float(getattr(config, "lr", optimizer.param_groups[0]["lr"]))
        num_cycles = int(getattr(config, "num_cycles", 10))
        return adjust_learning_rate_cos(optimizer, base_lr, epoch - 1, total_epochs, num_cycles)
    return float(optimizer.param_groups[0]["lr"])
