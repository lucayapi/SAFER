"""Export géométrie complète par fold SCGM → μ±σ K-fold / tuning."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.eval_geometry import evaluate_embeddings_geometry
from metrics.geometry import GEOMETRY_METRIC_KEYS
from safer_core.kfold_eval import aggregate_fold_rows
from scripts.train_scgm_text import _geometry_keys_from_row


def test_geometry_keys_from_row_covers_all_metric_keys():
    row = evaluate_embeddings_geometry(
        np.random.default_rng(0).standard_normal((16, 8)),
        np.array(["A0", "A1", "B", "C"] * 4),
        method="val",
    )
    keys = _geometry_keys_from_row(row)
    assert set(keys.keys()) == set(GEOMETRY_METRIC_KEYS)
    assert np.isfinite(keys["eta2_macro_balanced"])


def test_aggregate_fold_rows_std_from_geometry_rows():
    fold_rows = []
    for i in range(3):
        row = evaluate_embeddings_geometry(
            np.random.default_rng(i).standard_normal((12, 8)),
            np.array(["A0", "A1", "B", "C"] * 3),
            method="val",
        )
        fold_rows.append({"fold_id": i, **_geometry_keys_from_row(row)})
    agg = aggregate_fold_rows(fold_rows)
    assert agg["n_folds"] == 3
    for key in GEOMETRY_METRIC_KEYS:
        assert f"mean_{key}" in agg
        assert f"std_{key}" in agg
    assert agg["std_eta2_macro_balanced_perc"] >= 0
    assert agg["selection_score"] == agg["mean_eta2_macro_balanced_perc"]


def test_kfold_summary_row_has_std_columns_for_tuning_grid():
    """Simule export SCGM → colonnes attendues dans grid_summary (tuning)."""
    import pandas as pd

    fold_rows = []
    for i in range(2):
        row = evaluate_embeddings_geometry(
            np.random.default_rng(i + 1).standard_normal((10, 8)),
            np.array(["A0", "A1", "B", "C", "A0", "A1", "B", "C", "A0", "A1"]),
            method="val",
        )
        fold_rows.append({"fold_id": i, **_geometry_keys_from_row(row)})
    summary = aggregate_fold_rows(fold_rows)
    grid_row = {"combo_id": "test", "selection_score": summary["selection_score"], **summary}
    df = pd.DataFrame([grid_row])
    assert "std_eta2_macro_balanced_perc" in df.columns
    assert "mean_eta2_macro_balanced" in df.columns
