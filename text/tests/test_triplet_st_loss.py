"""Tests factory BatchHardSoftMarginTripletLoss (SBERT)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

pytest.importorskip("sentence_transformers")

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.losses.triplet_st import (
    build_batch_hard_soft_margin_loss,
    build_batch_triplet_loss,
)
from contrastive_methods.st_common import resolve_triplet_distance
from sentence_transformers import losses


def _minimal_cfg(**kwargs) -> ContrastiveConfig:
    base = dict(
        method_name="batch_triplet",
        dataset_path=TEXT_ROOT / "dataset/data_btp.csv",
        distance_metric="euclidean",
    )
    base.update(kwargs)
    return ContrastiveConfig(**base)


def test_resolve_triplet_distance_cosine_euclidean():
    assert resolve_triplet_distance("euclidean") is not None
    assert resolve_triplet_distance("cosine") is not None


def test_build_batch_hard_soft_margin_loss_type():
    cfg = _minimal_cfg()
    model = MagicMock()
    loss = build_batch_hard_soft_margin_loss(cfg, model)
    assert isinstance(loss, losses.BatchHardSoftMarginTripletLoss)


def test_build_batch_triplet_loss_default_native():
    cfg = _minimal_cfg()
    model = MagicMock()
    loss = build_batch_triplet_loss(cfg, model)
    assert isinstance(loss, losses.BatchHardSoftMarginTripletLoss)


def test_build_batch_hard_soft_margin_passes_margin_when_supported():
    cfg = _minimal_cfg(batch_triplet_margin=5.0)
    model = MagicMock()
    with patch.object(losses, "BatchHardSoftMarginTripletLoss") as mock_cls:
        mock_cls.return_value = MagicMock()
        build_batch_hard_soft_margin_loss(cfg, model)
        _, call_kwargs = mock_cls.call_args
        if "triplet_margin" in __import__("inspect").signature(
            losses.BatchHardSoftMarginTripletLoss.__init__
        ).parameters:
            assert call_kwargs.get("triplet_margin") == 5.0
