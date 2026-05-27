"""Optimizer factory for SCGM end-to-end training (AdamW, 3 param groups)."""

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


def _log_param_group(name: str, params: List[torch.nn.Parameter], lr: float | None = None) -> None:
    if not params:
        print(f"[SCGM Optimizer] group={name} skipped (0 trainable params)", flush=True)
        return
    n = sum(p.numel() for p in params)
    if lr is not None:
        print(
            f"[SCGM Optimizer] group={name} params={len(params)} numel={n:,} lr={lr}",
            flush=True,
        )
    else:
        print(
            f"[SCGM Optimizer] group={name} params={len(params)} numel={n:,}",
            flush=True,
        )


def build_optimizer(model: torch.nn.Module, config: Any) -> torch.optim.Optimizer:
    legacy = str(getattr(config, "optimizer", "adamw")).strip().lower()
    if legacy == "sgd":
        raise ValueError(
            "optimizer=sgd n'est plus supporté. SCGM utilise uniquement AdamW."
        )
    if legacy not in ("adamw", ""):
        raise ValueError(
            f"SCGM training only supports optimizer=adamw (got {legacy!r})."
        )

    lr_backbone = _lr(config, "lr_backbone", "backbone_lr", 5e-6)
    lr_projector = _lr(config, "lr_projector", "projector_lr", 5e-4)
    lr_head = _lr(config, "lr_head", "head_lr", 1e-3)

    wd_backbone = _wd(config, "weight_decay_backbone", "backbone_weight_decay", 1e-4)
    wd_projector = _wd(config, "weight_decay_projector", "projector_weight_decay", 1e-4)
    wd_head = _wd(config, "weight_decay_head", "head_weight_decay", 1e-4)

    param_groups: List[Dict[str, Any]] = []

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    _log_param_group("backbone", backbone_params, lr_backbone if backbone_params else None)
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
    _log_param_group("projector", proj_params, lr_projector)
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
    _log_param_group("head", scgm_params, lr_head)
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

    adam_beta1 = float(getattr(config, "adam_beta1", 0.9))
    adam_beta2 = float(getattr(config, "adam_beta2", 0.999))
    adam_eps = float(getattr(config, "adam_eps", 1e-8))
    print(
        f"[SCGM Optimizer] optimizer=adamw betas=({adam_beta1}, {adam_beta2}) eps={adam_eps}",
        flush=True,
    )
    return torch.optim.AdamW(
        param_groups,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
    )
