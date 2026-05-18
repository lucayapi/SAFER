"""Tests unitaires losses contrastives (CPU, petit tenseur)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import torch

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.losses.softtriple import SoftTripleLoss
from contrastive_methods.losses.supcon import SupConLoss


def test_softtriple_forward_shape():
    loss_mod = SoftTripleLoss(
        embedding_dim=8,
        num_classes=4,
        centers_per_class=2,
        tau=0.01,
        distance_metric="euclidean",
    )
    z = torch.randn(6, 8)
    labels = torch.tensor([0, 1, 2, 3, 0, 1])
    loss, stats = loss_mod(z, labels)
    assert loss.ndim == 0
    assert "loss_total" in stats


def test_supcon_loss_with_mock_model():
    emb = torch.randn(4, 16)
    model = MagicMock(side_effect=lambda _x: {"sentence_embedding": emb})
    loss_mod = SupConLoss(model=model, temperature=0.07, distance_metric="euclidean")
    features = ({"input_ids": torch.zeros(4, 3)},)
    labels = torch.tensor([0, 0, 1, 1])
    out = loss_mod(features, labels)
    assert out.ndim == 0
