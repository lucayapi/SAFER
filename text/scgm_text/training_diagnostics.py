"""Diagnostics SCGM end2end (paramètres entraînables, gradients)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


END2END_BANNER = """\
Running SCGM-Text in END2END mode (ICLR 2022-like).
Text -> backbone f_theta -> projector E_psi -> SCGM head (theta, psi, phi trainable).
No precomputed embeddings; no self-distillation on the main path.
"""


def describe_fidelity_mode(_args: Any) -> str:
    return END2END_BANNER


def _count_params(module: Optional[nn.Module]) -> Tuple[int, int]:
    if module is None:
        return 0, 0
    total = trainable = 0
    for p in module.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def print_end2end_startup(model: nn.Module) -> None:
    print("[SCGM] mode=end2end_text", flush=True)
    print("[SCGM] no precomputed embeddings", flush=True)
    bb_tr = any(p.requires_grad for p in model.backbone.parameters())
    pr_tr = any(p.requires_grad for p in model.projector.parameters())
    hd_tr = any(p.requires_grad for p in model.head.parameters())
    print(f"[SCGM] backbone trainable={bb_tr}", flush=True)
    print(f"[SCGM] projector trainable={pr_tr}", flush=True)
    print(f"[SCGM] head trainable={hd_tr}", flush=True)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[SCGM DEBUG] total params: {total:,}", flush=True)
    print(f"[SCGM DEBUG] trainable params: {trainable:,}", flush=True)

    for name, p in model.backbone.named_parameters():
        print(f"[SCGM DEBUG] backbone sample param: {name} requires_grad={p.requires_grad}", flush=True)
        break


def print_trainable_parameters(model: nn.Module) -> None:
    bb_tot, bb_tr = _count_params(model.backbone)
    pr_tot, pr_tr = _count_params(model.projector)
    scgm_tot, scgm_tr = _count_params(model.head)
    total = bb_tot + pr_tot + scgm_tot
    trainable = bb_tr + pr_tr + scgm_tr
    print(f"Paramètres totaux : {total:,}", flush=True)
    print(f"Paramètres entraînables : {trainable:,}", flush=True)
    print(f"  backbone entraînables : {bb_tr:,} / {bb_tot:,}", flush=True)
    print(f"  projection entraînables : {pr_tr:,} / {pr_tot:,}", flush=True)
    print(f"  SCGM entraînables : {scgm_tr:,} / {scgm_tot:,}", flush=True)


def assert_end2end_trainable(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    if not any(p.requires_grad for p in model.backbone.parameters()):
        raise RuntimeError("Backbone must be trainable in end2end SCGM.")
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    bb_in = any(id(p) in opt_ids for p in model.backbone.parameters() if p.requires_grad)
    if not bb_in:
        raise RuntimeError("No trainable backbone parameter in optimizer.")


def grad_norm_for_module(module: nn.Module) -> float:
    sq = 0.0
    for p in module.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum().item())
    return sq**0.5


def grad_norm_by_group(model: nn.Module) -> Dict[str, float]:
    return {
        "backbone": grad_norm_for_module(model.backbone),
        "projector": grad_norm_for_module(model.projector),
        "head": grad_norm_for_module(model.head),
    }


def print_grad_norms(model: nn.Module) -> None:
    norms = grad_norm_by_group(model)
    for name, val in norms.items():
        print(f"[SCGM DEBUG] grad_norm_{name}={val:.6e}", flush=True)
    if norms["backbone"] <= 0.0:
        raise RuntimeError("grad_norm_backbone is zero — backbone not in the loss graph.")


def snapshot_backbone_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    snap: Dict[str, torch.Tensor] = {}
    for name, p in model.backbone.named_parameters():
        if p.requires_grad and p.ndim > 1:
            snap[name] = p.detach().clone()
    return snap


def measure_backbone_weight_change(
    model: nn.Module,
    before: Dict[str, torch.Tensor],
) -> float:
    if not before:
        return 0.0
    max_change = 0.0
    for name, p in model.backbone.named_parameters():
        if name not in before:
            continue
        delta = (p.detach() - before[name]).abs().max().item()
        max_change = max(max_change, float(delta))
    return max_change


def verify_backbone_updated(
    model: nn.Module,
    _cfg: Any,
    before: Dict[str, torch.Tensor],
    max_abs_change: float,
) -> None:
    print(f"Backbone max abs weight change: {max_abs_change:.6e}", flush=True)
    if max_abs_change <= 0.0 and before:
        raise RuntimeError("Backbone expected to update but max weight change is 0.")
