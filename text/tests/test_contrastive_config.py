"""Tests chargement config contrastive native."""

from __future__ import annotations

import sys
from pathlib import Path

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.config import load_contrastive_config, load_contrastive_config_from_dict, merge_config_dict
from contrastive_methods.tuning import expand_grid
from safer_core.io import load_yaml
from safer_core.paths import TEXT_ROOT as TEXT_PKG_ROOT


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


def _grid_cfg(method: str):
    spec = load_yaml(TEXT_PKG_ROOT / f"configs/tuning/{method}_grid.yaml")
    base = load_yaml(TEXT_PKG_ROOT / f"configs/methods/{method}.yaml")
    combos = expand_grid(spec.get("grid") or {})
    merged = merge_config_dict(base, combos[0])
    return load_contrastive_config_from_dict(method, merged)


def test_tuning_grid_supcon_merges_distance_and_temperature():
    cfg = _grid_cfg("supcon")
    assert cfg.distance_metric == "euclidean"
    assert cfg.supcon_temperature in (0.05, 0.07)


def test_tuning_grid_batch_triplet_merges_distance():
    cfg = _grid_cfg("batch_triplet")
    assert cfg.distance_metric == "euclidean"


def test_tuning_grid_softtriple_merges_method_params():
    cfg = _grid_cfg("softtriple")
    assert cfg.distance_metric == "euclidean"
    assert cfg.softtriple_gamma in (0.1, 0.2)
    assert cfg.softtriple_lambda in (5.0, 10.0)
    assert cfg.softtriple_delta == 0.01
