"""Optimizer factory for SCGM-Text training."""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def _head_lr(config: Any) -> float:
    if getattr(config, "head_lr", None) is not None:
        return float(config.head_lr)
    return float(getattr(config, "lr", 1e-3))


def _head_weight_decay(config: Any) -> float:
    if getattr(config, "head_weight_decay", None) is not None:
        return float(config.head_weight_decay)
    return float(getattr(config, "weight_decay", 1e-4))


def build_optimizer(model: torch.nn.Module, config: Any) -> torch.optim.Optimizer:
    name = str(getattr(config, "optimizer", "adamw")).strip().lower()
    backbone_lr = float(getattr(config, "backbone_lr", 2e-5))
    backbone_wd = float(getattr(config, "backbone_weight_decay", 0.01))
    head_lr = _head_lr(config)
    head_wd = _head_weight_decay(config)

    param_groups: List[Dict[str, Any]] = []

    if getattr(model, "has_trainable_backbone", False):
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        if backbone_params:
            param_groups.append(
                {
                    "params": backbone_params,
                    "lr": backbone_lr,
                    "weight_decay": backbone_wd,
                    "name": "backbone",
                }
            )

    if getattr(model, "has_projection", False):
        proj_params = [p for p in model.projector.parameters() if p.requires_grad]
        if proj_params:
            param_groups.append(
                {
                    "params": proj_params,
                    "lr": head_lr,
                    "weight_decay": head_wd,
                    "name": "projection",
                }
            )

    scgm_params = []
    if hasattr(model, "scgm_parameters"):
        scgm_params = [p for p in model.scgm_parameters() if p.requires_grad]
    else:
        scgm_params = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and "projector" not in n and "backbone" not in n
        ]
    if scgm_params:
        param_groups.append(
            {
                "params": scgm_params,
                "lr": head_lr,
                "weight_decay": head_wd,
                "name": "scgm",
            }
        )

    if not param_groups:
        raise ValueError("Aucun paramètre entraînable pour l'optimiseur.")

    if name == "adamw":
        return torch.optim.AdamW(param_groups)
    if name == "sgd":
        momentum = float(getattr(config, "momentum", 0.9))
        return torch.optim.SGD(param_groups, momentum=momentum)
    raise ValueError(f"Unknown optimizer: {name!r} (expected adamw or sgd)")
