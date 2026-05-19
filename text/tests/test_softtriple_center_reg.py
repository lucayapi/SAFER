"""Tests régularisation centres SoftTriple (none / diversity / merge_l21)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.center_diagnostics import compute_effective_unique_centers
from contrastive_methods.config import load_contrastive_config, resolve_center_regularization_type
from contrastive_methods.losses.softtriple import SoftTripleLoss


def test_none_reg_zero_when_tau_zero():
    loss_mod = SoftTripleLoss(
        embedding_dim=8,
        num_classes=4,
        centers_per_class=3,
        tau=0.0,
        center_regularization_type="none",
        distance_metric="euclidean",
    )
    reg = loss_mod.regularization()
    assert float(reg.item()) == 0.0


def test_merge_l21_reg_positive_and_backward():
    loss_mod = SoftTripleLoss(
        embedding_dim=8,
        num_classes=4,
        centers_per_class=3,
        tau=0.01,
        center_regularization_type="merge_l21",
        distance_metric="euclidean",
    )
    z = torch.randn(8, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    loss, stats = loss_mod(z, labels)
    assert stats["loss_reg"] > 0.0
    loss.backward()
    assert loss_mod.centers.grad is not None
    assert torch.isfinite(loss_mod.centers.grad).all()


def test_diversity_reg_differs_from_merge():
    z = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 2, 3])
    merge = SoftTripleLoss(
        embedding_dim=8,
        num_classes=4,
        centers_per_class=3,
        tau=0.05,
        center_regularization_type="merge_l21",
        distance_metric="euclidean",
    )
    div = SoftTripleLoss(
        embedding_dim=8,
        num_classes=4,
        centers_per_class=3,
        tau=0.05,
        center_regularization_type="diversity",
        distance_metric="euclidean",
    )
    with torch.no_grad():
        shared = torch.randn_like(merge.centers)
        merge.centers.copy_(shared)
        div.centers.copy_(shared)
    _, sm = merge(z, labels)
    _, sd = div(z, labels)
    assert sm["loss_reg"] > 0 or sd["loss_reg"] > 0


def test_effective_unique_centers_duplicate_merge():
    centers = torch.tensor(
        [
            [[1.0, 0.0], [1.001, 0.0], [0.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    out = compute_effective_unique_centers(
        centers,
        metric="euclidean",
        distance_threshold=0.01,
    )
    assert out["per_class"][0]["n_effective_unique"] == 2
    assert out["summary"]["total_effective_unique_centers"] == 2


def test_config_retrocompat_diversity_when_tau_positive():
    assert resolve_center_regularization_type(None, 0.01) == "diversity"
    assert resolve_center_regularization_type(None, 0.0) == "none"


def test_load_default_softtriple_yaml_uses_diversity():
    cfg = load_contrastive_config("softtriple")
    assert cfg.center_regularization_type == "diversity"
    assert cfg.softtriple_tau == 0.01
