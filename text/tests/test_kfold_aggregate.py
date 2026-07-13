"""Tests agrégation K-fold μ±σ."""

from __future__ import annotations

import sys
from pathlib import Path

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from safer_core.kfold_eval import aggregate_fold_rows, group_kfold_splits


def test_aggregate_fold_rows_mean_std():
    rows = [
        {"fold_id": 0, "val_balanced_accuracy": 0.6, "val_accuracy": 0.55},
        {"fold_id": 1, "val_balanced_accuracy": 0.8, "val_accuracy": 0.75},
    ]
    agg = aggregate_fold_rows(rows, selection_metric="val_balanced_accuracy")
    assert agg["n_folds"] == 2
    assert abs(agg["mean_val_balanced_accuracy"] - 0.7) < 1e-6
    assert agg["selection_score"] == agg["mean_val_balanced_accuracy"]
    assert agg["std_val_balanced_accuracy"] > 0


def test_group_kfold_splits():
    groups = ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"]
    splits = group_kfold_splits(groups, 5, seed=42)
    assert len(splits) == 5
    for train_idx, val_idx in splits:
        assert len(set(train_idx) & set(val_idx)) == 0
