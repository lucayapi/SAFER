"""Tests agrégation métriques test K-fold."""

from __future__ import annotations

import sys
from pathlib import Path

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from safer_core.kfold_eval import aggregate_fold_rows, extract_test_metric_rows


def test_extract_test_metric_rows():
    fold_rows = [
        {"fold_id": 0, "delta_macro_pct": 10.0, "test_delta_macro_pct": 5.0},
        {"fold_id": 1, "delta_macro_pct": 20.0, "test_delta_macro_pct": 15.0},
    ]
    test_rows = extract_test_metric_rows(fold_rows)
    assert len(test_rows) == 2
    assert test_rows[0]["delta_macro_pct"] == 5.0
    agg = aggregate_fold_rows(test_rows)
    assert abs(agg["mean_delta_macro_pct"] - 10.0) < 1e-6
