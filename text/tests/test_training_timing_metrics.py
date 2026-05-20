"""Temps d'exécution K-fold (μ±σ) et fit final."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from safer_core.kfold_eval import (
    KFOLD_AGGREGATE_METRIC_KEYS,
    aggregate_fold_rows,
    record_final_fit_wall_time,
    save_kfold_tables,
)


def test_aggregate_fold_rows_train_wall_time_mean_std():
    rows = [
        {"fold_id": 0, "eta2_macro_balanced_perc": 10.0, "train_wall_time_sec": 100.0},
        {"fold_id": 1, "eta2_macro_balanced_perc": 12.0, "train_wall_time_sec": 120.0},
    ]
    agg = aggregate_fold_rows(rows, metric_keys=KFOLD_AGGREGATE_METRIC_KEYS)
    assert abs(agg["mean_train_wall_time_sec"] - 110.0) < 1e-6
    assert agg["std_train_wall_time_sec"] > 0


def test_save_kfold_tables_and_final_fit_time(tmp_path):
    fold_rows = [
        {"fold_id": 0, "eta2_macro_balanced_perc": 5.0, "train_wall_time_sec": 60.0},
        {"fold_id": 1, "eta2_macro_balanced_perc": 6.0, "train_wall_time_sec": 80.0},
    ]
    metrics_dir = tmp_path / "metrics"
    save_kfold_tables(fold_rows, metrics_dir, final_fit_wall_time_sec=300.5)
    summary = pd.read_csv(metrics_dir / "kfold_summary.csv")
    assert "mean_train_wall_time_sec" in summary.columns
    assert "std_train_wall_time_sec" in summary.columns
    assert abs(float(summary["final_fit_wall_time_sec"].iloc[0]) - 300.5) < 1e-6
    per_fold = pd.read_csv(metrics_dir / "kfold_per_fold.csv")
    assert "train_wall_time_sec" in per_fold.columns


def test_record_final_fit_wall_time_updates_existing_summary(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame([{"n_folds": 2, "mean_train_wall_time_sec": 70.0}]).to_csv(
        metrics_dir / "kfold_summary.csv", index=False
    )
    record_final_fit_wall_time(metrics_dir, 450.0)
    summary = pd.read_csv(metrics_dir / "kfold_summary.csv")
    assert float(summary["final_fit_wall_time_sec"].iloc[0]) == 450.0
