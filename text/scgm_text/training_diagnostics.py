"""Diagnostics d'entraînement (paramètres entraînables, mise à jour backbone)."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def _count_params(module: Optional[nn.Module]) -> Tuple[int, int]:
    if module is None:
        return 0, 0
    total = 0
    trainable = 0
    for p in module.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def print_trainable_parameters(model: nn.Module) -> None:
    backbone = getattr(model, "backbone", None)
    projector = getattr(model, "projector", None)
    head = getattr(model, "head", None)

    bb_tot, bb_tr = _count_params(backbone)
    pr_tot, pr_tr = _count_params(projector if getattr(model, "has_projection", True) else None)
    if not getattr(model, "has_projection", True):
        pr_tot, pr_tr = 0, 0
    scgm_tot, scgm_tr = _count_params(head)

    total = bb_tot + pr_tot + scgm_tot
    trainable = bb_tr + pr_tr + scgm_tr

    print(f"Paramètres totaux : {total:,}", flush=True)
    print(f"Paramètres entraînables : {trainable:,}", flush=True)
    print(f"  backbone entraînables : {bb_tr:,} / {bb_tot:,}", flush=True)
    print(f"  projection entraînables : {pr_tr:,} / {pr_tot:,}", flush=True)
    print(f"  SCGM entraînables : {scgm_tr:,} / {scgm_tot:,}", flush=True)


def warn_identity_frozen_backbone(cfg: Any) -> None:
    input_mode = str(getattr(cfg, "input_mode", "")).lower()
    projection = str(getattr(cfg, "projection", "")).lower()
    freeze = bool(getattr(cfg, "freeze_backbone", False))
    if input_mode == "text" and projection == "identity" and freeze:
        warnings.warn(
            "projection=identity with freeze_backbone=True means no trainable "
            "representation layer before SCGM except SCGM centers/parameters.",
            stacklevel=2,
        )


def assert_backbone_trainable_when_identity_text(
    model: nn.Module,
    cfg: Any,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    input_mode = str(getattr(cfg, "input_mode", "")).lower()
    projection = str(getattr(cfg, "projection", "")).lower()
    freeze = bool(getattr(cfg, "freeze_backbone", False))
    if not (input_mode == "text" and projection == "identity" and not freeze):
        return

    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise RuntimeError("input_mode=text mais backbone absent du modèle.")

    trainable = [p for p in backbone.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(
            "Backbone attendu entraînable (text + identity + freeze_backbone=false) "
            "mais aucun paramètre backbone n'a requires_grad=True."
        )

    if optimizer is not None:
        opt_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        in_opt = any(id(p) in opt_ids for p in trainable)
        if not in_opt:
            raise RuntimeError(
                "Aucun paramètre backbone entraînable n'est inclus dans l'optimiseur."
            )


def snapshot_backbone_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return {}
    snap: Dict[str, torch.Tensor] = {}
    for name, p in backbone.named_parameters():
        if p.requires_grad and p.ndim > 1:
            snap[name] = p.detach().clone()
    return snap


def measure_backbone_weight_change(
    model: nn.Module,
    before: Dict[str, torch.Tensor],
) -> float:
    backbone = getattr(model, "backbone", None)
    if backbone is None or not before:
        return 0.0
    max_change = 0.0
    for name, p in backbone.named_parameters():
        if name not in before:
            continue
        delta = (p.detach() - before[name]).abs().max().item()
        max_change = max(max_change, float(delta))
    return max_change


def verify_backbone_updated(
    model: nn.Module,
    cfg: Any,
    before: Dict[str, torch.Tensor],
    max_abs_change: float,
) -> None:
    input_mode = str(getattr(cfg, "input_mode", "")).lower()
    projection = str(getattr(cfg, "projection", "")).lower()
    freeze = bool(getattr(cfg, "freeze_backbone", False))

    print(f"Backbone max abs weight change: {max_abs_change:.6e}", flush=True)

    if input_mode == "text" and projection == "identity" and not freeze:
        if max_abs_change <= 0.0:
            raise RuntimeError(
                "Backbone is expected to be fine-tuned in identity mode, "
                "but no backbone parameter changed."
            )
    if input_mode == "text" and freeze and max_abs_change > 0.0:
        pass
