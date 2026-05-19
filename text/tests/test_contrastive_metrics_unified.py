"""Cohérence métriques géométrie contrastives."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.eval_geometry import evaluate_embeddings_geometry
from contrastive_methods.training_log import geometry_row_to_val_columns
from metrics.geometry import GEOMETRY_METRIC_KEYS
from safer_core.kfold_eval import aggregate_fold_rows


def test_geometry_row_to_val_columns_covers_all_keys():
    row = evaluate_embeddings_geometry(
        np.random.default_rng(1).standard_normal((12, 8)),
        np.array(["A0", "A1", "B", "C"] * 3),
        method="test",
    )
    val_cols = geometry_row_to_val_columns(row)
    assert len(val_cols) == len(GEOMETRY_METRIC_KEYS)
    assert set(val_cols) == {f"val_{k}" for k in GEOMETRY_METRIC_KEYS}


def test_aggregate_fold_rows_mean_for_all_numeric_keys():
    fold_rows = [
        evaluate_embeddings_geometry(
            np.random.default_rng(i).standard_normal((12, 8)),
            np.array(["A0", "A1", "B", "C"] * 3),
            method="val",
        )
        for i in range(3)
    ]
    for i, row in enumerate(fold_rows):
        row["fold_id"] = i
    agg = aggregate_fold_rows(fold_rows)
    for key in GEOMETRY_METRIC_KEYS:
        assert f"mean_{key}" in agg
        assert f"std_{key}" in agg
    assert agg["selection_score"] == agg["mean_eta2_macro_balanced_perc"]


def test_evaluate_embeddings_geometry_deterministic():
    labels = np.array(["A0", "A1", "B", "C", "A0", "A1", "B", "C"])
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((8, 16))
    row_a = evaluate_embeddings_geometry(emb, labels, method="test")
    row_b = evaluate_embeddings_geometry(emb.copy(), labels, method="test")
    assert row_a["eta2_macro_balanced_perc"] == row_b["eta2_macro_balanced_perc"]
    assert row_a["eta2_macro_balanced"] == row_b["eta2_macro_balanced"]
    assert row_a["rankme_global"] == row_b["rankme_global"]
