"""Tests unitaires losses contrastives (CPU, petit tenseur)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.losses.softtriple import SoftTripleLoss
from contrastive_methods.losses.supcon import SupConLoss
from contrastive_methods.losses.supcon_hobbit import SupConEmbeddingLoss
from contrastive_methods.losses.triplet_st import BatchTripletEmbeddingLoss


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


def test_supcon_embedding_loss():
    loss_mod = SupConEmbeddingLoss(temperature=0.07, normalize_embeddings=True)
    emb = torch.randn(4, 16)
    labels = torch.tensor([0, 0, 1, 1])
    out = loss_mod(emb, labels)
    assert out.ndim == 0
    assert torch.isfinite(out)


def test_batch_triplet_embedding_loss():
    loss_mod = BatchTripletEmbeddingLoss(distance_metric="euclidean", soft_margin=True)
    emb = torch.randn(8, 16)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    out = loss_mod(emb, labels)
    assert out.ndim == 0
    assert torch.isfinite(out)

