"""Vérifie les trois modes backbone_trainable sur mini-batch (dummy ou Qwen)."""

from __future__ import annotations

import argparse
import os
import sys

import torch
from argparse import Namespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scgm_text.optimizers import build_optimizer
from scgm_text.scgm_text_model import SCGMTextModel
from scgm_text.training_diagnostics import (
    assert_scgm_trainability,
    grad_norm_by_group,
    print_end2end_startup,
)


def _batch(device: torch.device, n: int = 8) -> dict:
    return {
        "input_ids": torch.randint(1, 50, (n, 12), device=device),
        "attention_mask": torch.ones(n, 12, dtype=torch.long, device=device),
        "label_ids": torch.randint(0, 4, (n,), device=device),
        "indices": torch.arange(n, device=device),
    }


def _run_mode(
    *,
    label: str,
    backbone_trainable: bool,
    train_last_n_layers: int | None,
    backbone_name: str,
    device: torch.device,
) -> None:
    print(f"\n=== Mode {label} ===", flush=True)
    model = SCGMTextModel(
        hiddim=32,
        num_classes=4,
        num_subclasses=8,
        backbone_model_name_or_path=backbone_name,
        projection="linear",
        backbone_trainable=backbone_trainable,
        train_last_n_layers=train_last_n_layers,
    ).to(device)
    print_end2end_startup(model)

    cfg = Namespace(
        optimizer="adamw",
        lr_backbone=1e-5,
        lr_projector=1e-2,
        lr_head=1e-2,
        weight_decay_backbone=0.0,
        weight_decay_projector=0.0,
        weight_decay_head=0.0,
    )
    opt = build_optimizer(model, cfg)
    assert_scgm_trainability(model, opt, expect_backbone_trainable=backbone_trainable)

    opt_names = {g.get("name") for g in opt.param_groups}
    if backbone_trainable:
        assert "backbone" in opt_names
    else:
        assert "backbone" not in opt_names
        assert opt_names == {"projector", "head"}

    batch = _batch(device)
    q = torch.zeros(8, 8, device=device)
    q[torch.arange(8), torch.randint(0, 8, (8,), device=device)] = 1.0
    y = torch.zeros(8, 4, device=device)
    y[torch.arange(8), batch["label_ids"]] = 1.0

    model.train()
    features = model(batch)
    loss, *_ = model.loss(features, q, y, tau=0.1, alpha=0.5)
    opt.zero_grad()
    loss.backward()
    norms = grad_norm_by_group(model)

    if backbone_trainable:
        assert norms["backbone"] > 0.0, f"{label}: grad_norm_backbone must be > 0"
    else:
        assert norms["backbone"] == 0.0, f"{label}: grad_norm_backbone must be 0"

    assert norms["projector"] > 0.0
    assert norms["head"] > 0.0

    if backbone_name == "__test_dummy__" and train_last_n_layers == 2:
        layer_flags = [
            any(p.requires_grad for p in layer.parameters())
            for layer in model.backbone.model.layers
        ]
        assert layer_flags == [False, False, True, True], layer_flags

    if backbone_name == "__test_dummy__" and backbone_trainable and train_last_n_layers is None:
        assert all(p.requires_grad for p in model.backbone.parameters())

    print(f"[OK] {label}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--backbone-name",
        type=str,
        default="__test_dummy__",
        help="Use Qwen/Qwen3-Embedding-0.6B for manual cluster check.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    name = args.backbone_name

    _run_mode(
        label="frozen",
        backbone_trainable=False,
        train_last_n_layers=None,
        backbone_name=name,
        device=device,
    )
    _run_mode(
        label="last2",
        backbone_trainable=True,
        train_last_n_layers=2,
        backbone_name=name,
        device=device,
    )
    _run_mode(
        label="full",
        backbone_trainable=True,
        train_last_n_layers=None,
        backbone_name=name,
        device=device,
    )
    print("\n[OK] all trainability modes passed", flush=True)


if __name__ == "__main__":
    main()
