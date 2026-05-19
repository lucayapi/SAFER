"""Tests tuning contrastif (sans entraînement HF)."""

from __future__ import annotations

import sys
from pathlib import Path

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.config import merge_config_dict
from contrastive_methods.eval_geometry import selection_score
from contrastive_methods.tuning import combo_id_from_overrides, expand_grid


def test_expand_grid_cartesian():
    grid = {"training.lr": [1e-5, 2e-5], "training.bs": [8, 16]}
    combos = expand_grid(grid)
    assert len(combos) == 4


def test_combo_id_readable():
    cid = combo_id_from_overrides({"training.learning_rate": 2e-5, "training.batch_size": 16})
    assert "learning_rate" in cid or "2e-05" in cid.lower()


def test_merge_config_dotted():
    base = {"training": {"lr": 1e-5}, "model": {"x": 1}}
    merged = merge_config_dict(base, {"training.lr": 2e-5})
    assert merged["training"]["lr"] == 2e-5


def test_selection_score_nan():
    assert selection_score({"eta2_macro_balanced_perc": float("nan")}) == float("-inf")
    assert selection_score({"eta2_macro_balanced_perc": 42.0}) == 42.0
