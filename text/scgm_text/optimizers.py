"""Optimizer factory for SCGM-Text training."""

from __future__ import annotations

from typing import Any

import torch


def build_optimizer(model: torch.nn.Module, config: Any) -> torch.optim.Optimizer:
    name = str(getattr(config, "optimizer", "adamw")).strip().lower()
    lr = float(getattr(config, "lr", 1e-3))
    weight_decay = float(getattr(config, "weight_decay", 1e-4))

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(getattr(config, "momentum", 0.9))
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {name!r} (expected adamw or sgd)")
