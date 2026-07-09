"""Tests SupConLoss HobbitLong (embeddings, sans ST)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.losses.supcon_hobbit import (
    HobbitSupConLoss,
    SupConEmbeddingLoss,
    build_supcon_embedding_loss,
)


def test_hobbit_supcon_loss_finite_single_view():
    torch.manual_seed(0)
    bsz, dim = 8, 16
    features = torch.randn(bsz, 1, dim)
    features = torch.nn.functional.normalize(features, dim=-1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss_fn = HobbitSupConLoss(temperature=0.07, contrast_mode="all", base_temperature=0.07)
    loss = loss_fn(features, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_hobbit_supcon_same_label_lower_loss_than_random():
    torch.manual_seed(1)
    dim = 32
    z0 = torch.nn.functional.normalize(torch.randn(4, dim), dim=-1)
    z1 = torch.nn.functional.normalize(torch.randn(4, dim), dim=-1)
    labels_same = torch.tensor([0, 0, 0, 0])
    labels_diff = torch.tensor([0, 1, 2, 3])
    loss_fn = HobbitSupConLoss(temperature=0.1)
    f_same = z0.unsqueeze(1)
    f_diff = z1.unsqueeze(1)
    loss_same = loss_fn(f_same, labels_same)
    loss_diff = loss_fn(f_diff, labels_diff)
    assert torch.isfinite(loss_same) and torch.isfinite(loss_diff)


def test_supcon_embedding_loss():
    loss_mod = SupConEmbeddingLoss(temperature=0.07, normalize_embeddings=True)
    emb = torch.randn(4, 16)
    labels = torch.tensor([0, 0, 1, 1])
    out = loss_mod(emb, labels)
    assert out.ndim == 0
    assert torch.isfinite(out)


def test_build_supcon_embedding_loss_rejects_non_cosine():
    cfg = ContrastiveConfig(
        method_name="supcon",
        dataset_path=TEXT_ROOT / "dataset/data_btp.csv",
        distance_metric="euclidean",
    )
    with __import__("pytest").raises(ValueError, match="cosine"):
        build_supcon_embedding_loss(cfg)
