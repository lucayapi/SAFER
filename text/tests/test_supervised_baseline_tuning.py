"""Tests tuning baseline supervisée sklearn (07b)."""

from __future__ import annotations

import json

import pandas as pd

from macro_transfer.supervised_baseline_tuning import (
    build_tuned_registry_from_best_rows,
    combo_id,
    compare_default_vs_tuned_cv,
    expand_param_grid,
    normalize_param_overrides,
    select_best_row_per_model,
)


def test_expand_param_grid_product():
    combos = expand_param_grid({"C": [0.1, 1.0], "max_depth": [None, 20]})
    assert len(combos) == 4
    assert {"C": 0.1, "max_depth": None} in combos


def test_normalize_null_string():
    assert normalize_param_overrides({"max_depth": "null"})["max_depth"] is None
    assert normalize_param_overrides({"max_depth": "None"})["max_depth"] is None


def test_combo_id_stable():
    a = combo_id("xgboost", {"max_depth": 6, "learning_rate": 0.1})
    b = combo_id("xgboost", {"learning_rate": 0.1, "max_depth": 6})
    assert a == b
    assert a.startswith("xgboost_")


def test_select_best_row_per_model():
    grid = pd.DataFrame(
        [
            {
                "model": "logistic_regression",
                "combo_id": "a",
                "selection_score": 0.5,
                "best_params": '{"C": 0.1}',
            },
            {
                "model": "logistic_regression",
                "combo_id": "b",
                "selection_score": 0.7,
                "best_params": '{"C": 1.0}',
            },
            {
                "model": "xgboost",
                "combo_id": "c",
                "selection_score": 0.6,
                "best_params": '{"max_depth": 4}',
            },
        ]
    )
    best = select_best_row_per_model(grid)
    assert best["logistic_regression"]["combo_id"] == "b"
    assert best["xgboost"]["combo_id"] == "c"


def test_build_tuned_registry_merges_params():
    best = {
        "logistic_regression": {
            "best_params": json.dumps({"C": 10.0}),
        }
    }
    reg = build_tuned_registry_from_best_rows(best)
    assert reg["logistic_regression"]["params"]["C"] == 10.0
    assert reg["logistic_regression"]["params"]["class_weight"] == "balanced"
    assert reg["logistic_regression"]["use_scaler"] is True


def test_compare_default_vs_tuned_cv():
    default = pd.DataFrame(
        [
            {
                "model": "xgboost",
                "mean_balanced_accuracy": 0.60,
                "std_balanced_accuracy": 0.02,
            }
        ]
    )
    tuned = pd.DataFrame(
        [
            {
                "model": "xgboost",
                "mean_balanced_accuracy": 0.65,
                "std_balanced_accuracy": 0.01,
            }
        ]
    )
    cmp = compare_default_vs_tuned_cv(default, tuned)
    assert abs(float(cmp.iloc[0]["delta_ba"]) - 0.05) < 1e-9
