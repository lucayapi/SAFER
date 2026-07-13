"""Tests agrégation métriques CV K-fold."""

from __future__ import annotations

import sys
from pathlib import Path

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from safer_core.kfold_eval import aggregate_fold_rows


def test_aggregate_classification_fold_rows():
    fold_rows = [
        {"fold_id": 0, "val_balanced_accuracy": 0.6, "val_accuracy": 0.55},
        {"fold_id": 1, "val_balanced_accuracy": 0.8, "val_accuracy": 0.75},
    ]
    agg = aggregate_fold_rows(fold_rows, selection_metric="val_balanced_accuracy")
    assert abs(agg["mean_val_balanced_accuracy"] - 0.7) < 1e-6
    assert agg["selection_score"] == agg["mean_val_balanced_accuracy"]
