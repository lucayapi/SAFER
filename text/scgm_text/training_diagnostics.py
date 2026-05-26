"""Diagnostics SCGM end2end (paramètres entraînables, gradients)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


END2END_BANNER = """\
Running SCGM-Text in END2END mode (ICLR 2022-like).
Text -> backbone f_theta -> projector E_psi -> SCGM head.
Backbone theta trainability is controlled by backbone_trainable / train_last_n_layers.
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

    cfg_bb = bool(getattr(model, "backbone_trainable", True))
    cfg_layers = getattr(model, "train_last_n_layers", None)
    print(f"[SCGM] backbone_trainable_config={cfg_bb}", flush=True)
    print(f"[SCGM] train_last_n_layers={cfg_layers}", flush=True)

    total_layers = model.backbone.num_transformer_layers()
    if total_layers is not None:
        print(f"[SCGM] total backbone layers={total_layers}", flush=True)
    unfrozen = getattr(model.backbone, "_unfrozen_layer_count", None)
    if cfg_bb and cfg_layers is None:
        print("[SCGM] full backbone fine-tuning", flush=True)
    elif cfg_bb and cfg_layers is not None:
        trainable_layers = model.backbone.count_trainable_transformer_layers()
        print(f"[SCGM] unfrozen backbone layers={trainable_layers}", flush=True)

    bb_tr = model.has_trainable_backbone
    pr_tr = any(p.requires_grad for p in model.projector.parameters())
    hd_tr = any(p.requires_grad for p in model.head.parameters())
    print(f"[SCGM] backbone actual trainable={bb_tr}", flush=True)
    print(f"[SCGM] projector trainable={pr_tr}", flush=True)
    print(f"[SCGM] head trainable={hd_tr}", flush=True)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[SCGM DEBUG] total params: {total:,}", flush=True)
    print(f"[SCGM DEBUG] trainable params: {trainable:,}", flush=True)

    for name, p in model.backbone.named_parameters():
        if p.requires_grad:
            print(
                f"[SCGM DEBUG] backbone sample trainable param: {name}",
                flush=True,
            )
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


def assert_scgm_trainability(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expect_backbone_trainable: bool,
) -> None:
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}

    if not any(p.requires_grad for p in model.projector.parameters()):
        raise RuntimeError("Projector must have trainable parameters.")
    if not any(p.requires_grad for p in model.head.parameters()):
        raise RuntimeError("SCGM head must have trainable parameters.")
    if not any(
        id(p) in opt_ids for p in model.projector.parameters() if p.requires_grad
    ):
        raise RuntimeError("No trainable projector parameter in optimizer.")
    if not any(id(p) in opt_ids for p in model.head.parameters() if p.requires_grad):
        raise RuntimeError("No trainable SCGM head parameter in optimizer.")

    bb_trainable = model.has_trainable_backbone
    bb_in_opt = any(
        id(p) in opt_ids for p in model.backbone.parameters() if p.requires_grad
    )

    if expect_backbone_trainable:
        if not bb_trainable:
            raise RuntimeError(
                "backbone_trainable=true but no backbone parameter has requires_grad."
            )
        if not bb_in_opt:
            raise RuntimeError("No trainable backbone parameter in optimizer.")
    else:
        if bb_trainable:
            raise RuntimeError(
                "backbone_trainable=false but backbone still has requires_grad parameters."
            )
        if bb_in_opt:
            raise RuntimeError(
                "Frozen backbone must not appear in optimizer param groups."
            )


def assert_end2end_trainable(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Compat legacy : exige backbone entraînable."""
    assert_scgm_trainability(model, optimizer, expect_backbone_trainable=True)


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


def print_grad_norms(
    model: nn.Module,
    *,
    expect_backbone_trainable: bool = True,
) -> None:
    norms = grad_norm_by_group(model)
    for name, val in norms.items():
        print(f"[SCGM DEBUG] grad_norm_{name}={val:.6e}", flush=True)
    if expect_backbone_trainable:
        if norms["backbone"] <= 0.0:
            raise RuntimeError("grad_norm_backbone is zero — backbone not in the loss graph.")
    else:
        print(
            "[SCGM DEBUG] grad_norm_backbone=0.0 expected because backbone_trainable=false",
            flush=True,
        )
    if norms["projector"] <= 0.0:
        raise RuntimeError("grad_norm_projector is zero.")
    if norms["head"] <= 0.0:
        raise RuntimeError("grad_norm_head is zero.")


def snapshot_backbone_weights(
    model: nn.Module,
    *,
    all_params: bool = False,
) -> Dict[str, torch.Tensor]:
    snap: Dict[str, torch.Tensor] = {}
    for name, p in model.backbone.named_parameters():
        if all_params or p.requires_grad:
            if p.ndim > 0:
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
    cfg: Any,
    before: Dict[str, torch.Tensor],
    max_abs_change: float,
    *,
    tolerance: float = 1e-9,
) -> None:
    expect_trainable = bool(getattr(cfg, "backbone_trainable", True))
    print(f"Backbone max abs weight change: {max_abs_change:.6e}", flush=True)
    if expect_trainable:
        if before and max_abs_change <= tolerance:
            raise RuntimeError("Backbone expected to update but did not.")
    else:
        if max_abs_change > tolerance:
            raise RuntimeError(
                f"Backbone is frozen but weights changed (max_abs_change={max_abs_change:.6e})."
            )
        print("[SCGM DEBUG] backbone frozen: no weight change expected", flush=True)
