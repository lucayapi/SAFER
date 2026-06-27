"""Tests géométrie supervised_macro_ft."""

from __future__ import annotations

import numpy as np
import pandas as pd

from metrics.geometry import GEOMETRY_METRIC_KEYS
from metrics.intra_role_preservation import IPR_COLUMNS
from safer_core.kfold_eval import KFOLD_AGGREGATE_METRIC_KEYS, aggregate_fold_rows
from supervised_macro_ft.geometry_eval import (
    evaluate_projected_geometry,
    geometry_keys_from_row,
    save_geometry_kfold_tables,
)


def test_evaluate_projected_geometry_and_keys():
    z = np.random.default_rng(0).standard_normal((24, 16))
    labels = np.array(["A0", "A1", "B", "C"] * 6)
    row = evaluate_projected_geometry(z, labels, method="test")
    keys = geometry_keys_from_row(row)
    assert set(keys.keys()) == set(GEOMETRY_METRIC_KEYS)
    assert np.isfinite(keys["eta2_macro_balanced"])


def test_aggregate_geometry_fold_rows_mean_std():
    fold_rows = []
    for i in range(3):
        z = np.random.default_rng(i).standard_normal((20, 8))
        labels = np.array(["A0", "A1", "B", "C"] * 5)
        geom = geometry_keys_from_row(evaluate_projected_geometry(z, labels))
        ipr = {col: 0.9 + 0.05 * i for col in IPR_COLUMNS}
        fold_rows.append({"fold": i, **geom, **ipr})
    agg = aggregate_fold_rows(fold_rows, metric_keys=KFOLD_AGGREGATE_METRIC_KEYS)
    assert agg["n_folds"] == 3
    assert "mean_eta2_macro_balanced_perc" in agg
    assert "std_IPR_mean" in agg


def test_save_geometry_kfold_tables(tmp_path):
    z = np.random.default_rng(1).standard_normal((12, 8))
    labels = np.array(["A0", "A1", "B", "C"] * 3)
    geom = geometry_keys_from_row(evaluate_projected_geometry(z, labels))
    save_geometry_kfold_tables([{"fold": 0, **geom}], tmp_path)
    assert (tmp_path / "kfold_geometry_per_fold.csv").is_file()
    assert (tmp_path / "kfold_geometry_summary.csv").is_file()
    summary = pd.read_csv(tmp_path / "kfold_geometry_summary.csv")
    assert "mean_eta2_macro_balanced" in summary.columns
