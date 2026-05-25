"""Optimizer factory for SCGM strict-fidelity training (SGD, 3 param groups)."""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def _lr(config: Any, primary: str, fallback: str, default: float) -> float:
    val = getattr(config, primary, None)
    if val is not None:
        return float(val)
    return float(getattr(config, fallback, default))


def _wd(config: Any, primary: str, fallback: str, default: float) -> float:
    val = getattr(config, primary, None)
    if val is not None:
        return float(val)
    return float(getattr(config, fallback, default))


def build_optimizer(model: torch.nn.Module, config: Any) -> torch.optim.Optimizer:
    name = str(getattr(config, "optimizer", "sgd")).strip().lower()
    if name != "sgd":
        raise ValueError(
            f"SCGM training only supports optimizer=sgd (got {name!r}). "
            "Adam/AdamW have been removed."
        )

    lr_backbone = _lr(config, "lr_backbone", "backbone_lr", 1e-5)
    lr_projector = _lr(config, "lr_projector", "head_lr", 0.01)
    lr_head = _lr(config, "lr_head", "head_lr", 0.03)

    wd_backbone = _wd(config, "weight_decay_backbone", "backbone_weight_decay", 1e-4)
    wd_projector = _wd(config, "weight_decay_projector", "head_weight_decay", 1e-4)
    wd_head = _wd(config, "weight_decay_head", "head_weight_decay", 1e-4)

    param_groups: List[Dict[str, Any]] = []

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": lr_backbone,
                "weight_decay": wd_backbone,
                "name": "backbone",
            }
        )

    proj_params = [p for p in model.projector.parameters() if p.requires_grad]
    if proj_params:
        param_groups.append(
            {
                "params": proj_params,
                "lr": lr_projector,
                "weight_decay": wd_projector,
                "name": "projector",
            }
        )

    scgm_params = [p for p in model.scgm_parameters() if p.requires_grad]
    if scgm_params:
        param_groups.append(
            {
                "params": scgm_params,
                "lr": lr_head,
                "weight_decay": wd_head,
                "name": "head",
            }
        )

    if not param_groups:
        raise ValueError("Aucun paramètre entraînable pour l'optimiseur.")

    momentum = float(getattr(config, "momentum", 0.9))
    return torch.optim.SGD(param_groups, momentum=momentum)
