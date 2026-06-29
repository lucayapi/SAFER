"""Tests tuning supervised_macro_ft (grille, merge, mock CV)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.config import merge_config_dict
from supervised_macro_ft.tuning import (
    _combo_id,
    _merge_overrides,
    expand_grid,
    validate_macro_ft_grid_keys,
)


def test_validate_macro_ft_grid_keys_accepts_model_training():
    validate_macro_ft_grid_keys(
        {
            "model.projection": ["linear"],
            "training.lr_projector": [1e-3],
        }
    )


def test_validate_macro_ft_grid_keys_rejects_unknown():
    try:
        validate_macro_ft_grid_keys({"data.batch_size": [8]})
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "data.batch_size" in str(exc)


def test_expand_grid_cartesian():
    grid = {
        "model.projection": ["linear", "mlp_sklearn"],
        "training.lr_projector": [1e-3, 5e-4],
    }
    combos = expand_grid(grid)
    assert len(combos) == 4
    assert {"model.projection": "mlp_sklearn", "training.lr_projector": 5e-4} in combos


def test_merge_overrides_dotted_keys():
    base = {"model": {"projection": "linear", "hiddim": 512}, "training": {"lr_head": 1e-3}}
    merged = _merge_overrides(base, {"model.projection": "ln_gelu", "model.hiddim": 256})
    assert merged["model"]["projection"] == "ln_gelu"
    assert merged["model"]["hiddim"] == 256
    assert merged["training"]["lr_head"] == 1e-3


def test_combo_id_readable():
    cid = _combo_id({"model.projection": "mlp_sklearn", "training.lr_projector": 5e-4})
    assert "projection" in cid
    assert "mlp_sklearn" in cid


@patch("supervised_macro_ft.tuning.run_supervised_macro_ft_cv")
def test_run_combo_cv_selection_score_from_mean_balanced_accuracy(mock_cv):
    from supervised_macro_ft.tuning import _run_combo_cv

    mock_cv.return_value = {
        "mean_balanced_accuracy": 0.72,
        "std_balanced_accuracy": 0.03,
        "n_folds": 5,
    }
    base_cfg = merge_config_dict(
        {"model": {"projection": "linear"}, "training": {"seed": 42}},
        {},
    )
    row = _run_combo_cv(
        base_cfg,
        {"model.projection": "linear"},
        combo_output_dir=Path("/tmp/combo"),
        backbone_hidden=None,
        shared_cache_dir=Path("/tmp/cache"),
        n_folds=5,
        seed=42,
        selection_metric="balanced_accuracy",
    )
    assert row["selection_score"] == 0.72
    assert row["mean_balanced_accuracy"] == 0.72


def test_grid_yaml_loads():
    from safer_core.io import load_yaml

    spec = load_yaml(TEXT_ROOT / "configs/tuning/supervised_macro_ft_grid.yaml")
    assert spec["selection_metric"] == "balanced_accuracy"
    assert "mlp_sklearn" in spec["grid"]["model.projection"]
    combos = expand_grid(spec["grid"])
    assert len(combos) == 96
