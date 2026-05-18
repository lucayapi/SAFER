"""Tests chargement config contrastive native."""

from __future__ import annotations

import sys
from pathlib import Path

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.config import load_contrastive_config


def test_load_batch_triplet_yaml():
    cfg = load_contrastive_config("batch_triplet")
    assert cfg.method_name == "batch_triplet"
    assert cfg.val_ratio == 0.1
    assert cfg.batch_size == 16
    assert cfg.distance_metric == "euclidean"


def test_load_softtriple_yaml():
    cfg = load_contrastive_config("softtriple")
    assert cfg.centers_per_class == 5
    assert cfg.softtriple_lambda == 10.0


def test_load_supcon_yaml():
    cfg = load_contrastive_config("supcon")
    assert cfg.supcon_temperature == 0.07
    assert cfg.supcon_normalize_embeddings is True
    assert cfg.distance_metric == "euclidean"


def test_load_softtriple_distance_euclidean():
    cfg = load_contrastive_config("softtriple")
    assert cfg.distance_metric == "euclidean"
    assert cfg.center_min_distance == 0.3
