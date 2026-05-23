"""Tests diagnostics Batch Hard Triplet (torch pur, sans entraînement ST)."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("sentence_transformers")

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.losses.triplet_st import (
    BatchTripletLossWithDiagnostics,
    build_batch_triplet_loss,
)
from contrastive_methods.triplet_diagnostics import (
    TRIPLET_DIAG_CSV_COLUMNS,
    TripletDiagnosticLogger,
    compute_batch_hard_triplet_stats,
    pairwise_distance_matrix,
    triplet_loss_from_hard_distances,
)


def test_pairwise_cosine_symmetric_zero_diag():
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    d = pairwise_distance_matrix(z, "cosine")
    assert d.shape == (3, 3)
    assert torch.allclose(d, d.T, atol=1e-5)
    assert torch.allclose(torch.diag(d), torch.zeros(3), atol=1e-5)


def test_pairwise_euclidean_triangle_inequality():
    x = torch.randn(5, 8)
    d = pairwise_distance_matrix(x, "euclidean")
    for i in range(5):
        for j in range(5):
            for k in range(5):
                assert d[i, j] <= d[i, k] + d[k, j] + 1e-4


def test_batch_hard_stats_positive_gap():
    emb = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [-1.0, 0.0],
            [-0.9, 0.1],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1])
    stats = compute_batch_hard_triplet_stats(
        emb, labels, distance_metric="cosine", soft_margin=True, margin=None, eps=1e-6
    )
    assert stats.n_valid_anchors == 4
    assert stats.mean_hard_neg_dist > stats.mean_hard_pos_dist
    assert stats.triplet_gap > 0


def test_batch_no_valid_anchors_nan():
    emb = torch.randn(3, 4)
    labels = torch.tensor([0, 1, 2])
    stats = compute_batch_hard_triplet_stats(emb, labels, distance_metric="cosine")
    assert stats.n_valid_anchors == 0
    assert math.isnan(stats.mean_hard_pos_dist)
    assert math.isnan(stats.triplet_gap)


def test_active_ratio_soft_vs_hard():
    emb = torch.tensor([[1.0, 0], [0.9, 0.1], [-1.0, 0], [-0.9, 0.1]])
    labels = torch.tensor([0, 0, 1, 1])
    s_soft = compute_batch_hard_triplet_stats(
        emb, labels, distance_metric="cosine", soft_margin=True, eps=1e-6
    )
    s_hard = compute_batch_hard_triplet_stats(
        emb,
        labels,
        distance_metric="cosine",
        soft_margin=False,
        margin=0.1,
        eps=1e-6,
    )
    assert 0 <= s_soft.active_triplet_ratio <= 1
    assert 0 <= s_hard.active_triplet_ratio <= 1


def test_triplet_loss_from_hard_distances_grad():
    d_pos = torch.tensor([0.2, 0.3], requires_grad=True)
    d_neg = torch.tensor([0.5, 0.6], requires_grad=True)
    loss = triplet_loss_from_hard_distances(d_pos, d_neg, soft_margin=True)
    loss.backward()
    assert d_pos.grad is not None


def test_diagnostic_logger_csv(tmp_path: Path):
    path = tmp_path / "batch_triplet_diagnostics.csv"
    logger = TripletDiagnosticLogger(path, every_steps=1)
    emb = torch.tensor([[1.0, 0], [0.9, 0.1], [-1.0, 0], [-0.9, 0.1]])
    labels = torch.tensor([0, 0, 1, 1])
    stats = compute_batch_hard_triplet_stats(emb, labels, distance_metric="cosine")
    logger.maybe_log(stats, 0.69)
    logger.maybe_log(stats, 0.68)
    assert path.exists()
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert list(rows[0].keys()) == TRIPLET_DIAG_CSV_COLUMNS


def test_build_batch_triplet_loss_native_vs_custom():
    cfg_off = ContrastiveConfig(
        method_name="batch_triplet",
        dataset_path=Path("."),
        triplet_log_diagnostics=False,
        triplet_implementation="sentence_transformers",
    )
    model = MagicMock()
    loss_off = build_batch_triplet_loss(cfg_off, model)
    from sentence_transformers import losses as st_losses

    assert isinstance(loss_off, st_losses.BatchHardSoftMarginTripletLoss)

    cfg_on = ContrastiveConfig(
        method_name="batch_triplet",
        dataset_path=Path("."),
        triplet_log_diagnostics=True,
        triplet_implementation="custom_diagnostics",
        distance_metric="cosine",
    )
    loss_on = build_batch_triplet_loss(cfg_on, model)
    assert isinstance(loss_on, BatchTripletLossWithDiagnostics)


def test_wrapper_forward_requires_grad():
    model = MagicMock()

    def encode_forward(features):
        n = features["input_ids"].shape[0]
        return {"sentence_embedding": torch.randn(n, 4, requires_grad=True)}

    model.side_effect = encode_forward

    wrapper = BatchTripletLossWithDiagnostics(
        model=model,
        distance_metric="cosine",
        soft_margin=True,
        margin=None,
        diagnostic_logger=None,
    )
    labels = torch.tensor([0, 0, 1, 1])
    features = [{"input_ids": torch.zeros(4, 2, dtype=torch.long)}]
    loss = wrapper(features, labels)
    assert loss.ndim == 0
    loss.backward()
