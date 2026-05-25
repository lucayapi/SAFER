"""Learning-rate schedulers for SCGM end2end (cosine per param group)."""

from __future__ import annotations

import math
from typing import Any, List, Optional


def _store_base_lrs(optimizer) -> None:
    for group in optimizer.param_groups:
        if "base_lr" not in group:
            group["base_lr"] = float(group["lr"])


def adjust_learning_rate_cos_multi(
    optimizer,
    epoch: int,
    num_epochs: int,
    num_cycles: int,
) -> float:
    """Cosine schedule; each param group scales from its own base_lr."""
    epochs_per_cycle = max(1, math.floor(num_epochs / max(1, num_cycles)))
    phase = (epoch % epochs_per_cycle) / epochs_per_cycle
    scale = 0.5 * (1.0 + math.cos(math.pi * phase))
    first_lr = None
    for group in optimizer.param_groups:
        base = float(group.get("base_lr", group["lr"]))
        new_lr = base * scale
        group["lr"] = new_lr
        if first_lr is None:
            first_lr = new_lr
    return float(first_lr or 0.0)


def build_scheduler(optimizer, config: Any) -> Optional[str]:
    name = str(getattr(config, "scheduler", "none")).strip().lower()
    if name in ("none", ""):
        return None
    if name in ("cosine", "cosine_warm_restarts"):
        _store_base_lrs(optimizer)
        return "cosine"
    raise ValueError(f"Unknown scheduler: {name!r} (expected none, cosine, cosine_warm_restarts)")


def step_scheduler(
    optimizer,
    config: Any,
    epoch: int,
    total_epochs: int,
) -> float:
    scheduler = build_scheduler(optimizer, config)
    if scheduler is None:
        return float(optimizer.param_groups[0]["lr"])
    if scheduler == "cosine":
        num_cycles = int(getattr(config, "num_cycles", 10))
        return adjust_learning_rate_cos_multi(optimizer, epoch - 1, total_epochs, num_cycles)
    return float(optimizer.param_groups[0]["lr"])
