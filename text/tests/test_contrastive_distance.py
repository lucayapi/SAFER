"""Tests distance euclidienne partagée (contrastive)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.distance import (
    embedding_to_center_scores,
    normalize_distance_metric,
    pairwise_logits,
)
from contrastive_methods.losses.softtriple import SoftTripleLoss
from contrastive_methods.losses.supcon import SupConLoss
from unittest.mock import MagicMock


def test_normalize_distance_metric_default():
    assert normalize_distance_metric("") == "euclidean"
    assert normalize_distance_metric("eucledian") == "euclidean"


def test_pairwise_logits_euclidean_vs_cosine():
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    z = torch.nn.functional.normalize(z, p=2, dim=1)
    eucl = pairwise_logits(z, metric="euclidean", temperature=1.0)
    cos = pairwise_logits(z, metric="cosine", temperature=1.0)
    assert eucl.shape == (3, 3)
    assert not torch.allclose(eucl, cos)
    assert eucl[0, 2] > eucl[0, 1]  # même classe (0) plus proche que classe 1


def test_embedding_to_center_scores_euclidean():
    z = torch.randn(4, 8)
    centers = torch.randn(3, 2, 8)
    scores = embedding_to_center_scores(z, centers, metric="euclidean")
    assert scores.shape == (4, 3, 2)
    assert scores.max() <= 0.0


def test_softtriple_euclidean_forward():
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
    assert loss_mod.distance_metric == "euclidean"
    assert not loss_mod.normalize_embeddings
    assert "loss_total" in stats


def test_supcon_euclidean_forward():
    emb = torch.randn(4, 16)
    model = MagicMock(side_effect=lambda _x: {"sentence_embedding": emb})
    loss_mod = SupConLoss(
        model=model,
        temperature=0.07,
        distance_metric="euclidean",
    )
    out = loss_mod(({"input_ids": torch.zeros(4, 3)},), torch.tensor([0, 0, 1, 1]))
    assert out.ndim == 0
    assert loss_mod.distance_metric == "euclidean"
