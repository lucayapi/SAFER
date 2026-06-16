"""Tests IPR agrégé en validation K-fold."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.eval_geometry import compute_fold_ipr, evaluate_embeddings_geometry
from metrics.intra_role_preservation import IPR_COLUMNS, compute_ipr_from_geometry_rows
from safer_core.kfold_eval import KFOLD_AGGREGATE_METRIC_KEYS, aggregate_fold_rows
from scripts.train_scgm_text import _geometry_keys_from_row


def test_kfold_aggregate_includes_ipr_keys():
    assert "IPR_mean" in KFOLD_AGGREGATE_METRIC_KEYS
    assert "IPR_A0" in KFOLD_AGGREGATE_METRIC_KEYS


def test_aggregate_fold_rows_ipr_mean_std():
    fold_rows = []
    for i, w_a0 in enumerate([4.0, 3.5, 5.0]):
        method = evaluate_embeddings_geometry(
            np.random.default_rng(i).standard_normal((12, 8)),
            np.array(["A0", "A1", "B", "C"] * 3),
            method="val",
        )
        raw = dict(method)
        raw["W_A0"] = 2.0
        ipr = compute_ipr_from_geometry_rows(raw, method)
        fold_rows.append({"fold_id": i, **_geometry_keys_from_row(method), **ipr})
    agg = aggregate_fold_rows(fold_rows, metric_keys=KFOLD_AGGREGATE_METRIC_KEYS)
    assert agg["n_folds"] == 3
    assert "mean_IPR_mean" in agg
    assert "std_IPR_A0" in agg


def test_compute_fold_ipr_missing_emb_returns_nan(tmp_path):
    val_df = pd.DataFrame(
        {
            "doc_id": [1, 2, 3, 4],
            "pred_label": ["A0", "A1", "B", "C"],
        }
    )
    method_geom = {
        "T_macro_balanced": 10.0,
        "W_A0": 4.0,
        "W_A1": 2.0,
        "W_B": 2.0,
        "W_C": 2.0,
    }
    ipr = compute_fold_ipr(val_df, "pred_label", method_geom, emb_csv=tmp_path / "missing.csv")
    assert set(ipr.keys()) == set(IPR_COLUMNS)
    assert all(np.isnan(v) for v in ipr.values())
