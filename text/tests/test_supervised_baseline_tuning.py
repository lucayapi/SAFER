"""Tests tuning baseline supervisée sklearn (07b)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from macro_transfer.supervised_baseline_tuning import (
    build_tuned_registry_from_best_rows,
    combo_id,
    compare_default_vs_tuned_cv,
    expand_param_grid,
    export_final_results_table,
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


def test_export_final_results_table(tmp_path):
    best = {
        "logistic_regression": {
            "combo_id": "lr1",
            "selection_score": 0.80,
            "std_balanced_accuracy": 0.01,
            "mean_accuracy": 0.85,
            "best_params": '{"C": 1.0}',
        },
        "xgboost": {
            "combo_id": "xgb1",
            "selection_score": 0.82,
            "std_balanced_accuracy": 0.02,
            "mean_accuracy": 0.87,
            "best_params": {"max_depth": 6, "learning_rate": 0.1},
        },
    }
    ood = {
        "metallurgie": {"logistic_regression": 0.70, "xgboost": 0.72},
        "caou": {"logistic_regression": 0.65, "xgboost": 0.68},
    }
    summary = export_final_results_table(
        tmp_path,
        best,
        best_model="xgboost",
        ood_ba_by_corpus=ood,
    )
    assert (tmp_path / "results_summary.csv").is_file()
    assert (tmp_path / "best_hyperparams.json").is_file()
    assert list(summary["model"])[0] == "xgboost"
    assert bool(summary.loc[summary["model"] == "xgboost", "is_best_overall"].iloc[0])
    assert summary.loc[summary["model"] == "xgboost", "ba_ood_avg"].iloc[0] == pytest.approx(0.70)


def test_medium_grid_sizes_match_yaml_intent():
    """Garde-fou : tailles de grilles « moyennes » du YAML de prod."""
    lr = expand_param_grid({"C": [0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0]})
    rf = expand_param_grid(
        {
            "n_estimators": [200, 400, 800],
            "max_depth": [None, 20, 40],
            "min_samples_leaf": [1, 4],
        }
    )
    xgb = expand_param_grid(
        {
            "n_estimators": [150, 300, 500],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.03, 0.1],
            "subsample": [0.8, 1.0],
        }
    )
    assert len(lr) == 8
    assert len(rf) == 18
    assert len(xgb) == 36
    assert len(lr) + len(rf) + len(xgb) == 62
