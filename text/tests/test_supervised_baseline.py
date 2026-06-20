"""Tests légers pour macro_transfer.supervised_baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macro_transfer.constants import MACRO_NAMES
from macro_transfer.frozen_source_prototypes import _build_gating_from_predictions
from macro_transfer.supervised_baseline import (
    aggregate_cv_metrics,
    build_classifier_pipeline,
    build_predictions_dataframe,
    merge_model_registry,
    run_model_group_kfold_cv,
    select_best_model,
    _fit_pipeline,
    _resample_train_for_class_weight,
)
from safer_core.kfold_eval import group_kfold_splits

TEXT_ROOT = Path(__file__).resolve().parents[1]


def test_aggregate_cv_metrics_three_folds():
    rows = []
    for fold_id in range(3):
        rows.append(
            {
                "model": "logistic_regression",
                "fold_id": fold_id,
                "accuracy": 0.6 + 0.05 * fold_id,
                "macro_f1": 0.5 + 0.04 * fold_id,
                "balanced_accuracy": 0.55 + 0.03 * fold_id,
            }
        )
    summary = aggregate_cv_metrics(rows)
    assert len(summary) == 1
    assert summary.loc[0, "model"] == "logistic_regression"
    assert summary.loc[0, "n_folds"] == 3
    assert abs(summary.loc[0, "mean_accuracy"] - 0.65) < 1e-9
    assert summary.loc[0, "std_macro_f1"] > 0


def test_group_kfold_no_leakage():
    groups = np.array(["a1", "a1", "a2", "a2", "a3", "a3", "a4", "a4"])
    splits = group_kfold_splits(groups, n_splits=2, seed=42)
    for tr_idx, va_idx in splits:
        tr_groups = set(groups[tr_idx])
        va_groups = set(groups[va_idx])
        assert tr_groups.isdisjoint(va_groups)


def test_build_gating_from_classifier_probs():
    preds = pd.DataFrame(
        {
            "pred_macro": ["A0", "B"],
            "confidence": [0.9, 0.8],
            "prob_A0": [0.9, 0.1],
            "prob_A1": [0.05, 0.1],
            "prob_B": [0.03, 0.7],
            "prob_C": [0.02, 0.1],
        }
    )
    gating = _build_gating_from_predictions(preds, MACRO_NAMES)
    assert list(gating["m_hat"]) == ["A0", "B"]
    assert float(gating.loc[0, "prob_A0"]) == pytest.approx(0.9)
    assert float(gating.loc[1, "prob_B"]) == pytest.approx(0.7)


def test_select_best_model_by_macro_f1():
    summary = pd.DataFrame(
        [
            {"model": "a", "mean_macro_f1": 0.4},
            {"model": "b", "mean_macro_f1": 0.55},
        ]
    )
    assert select_best_model(summary, selection_metric="macro_f1") == "b"


def test_build_classifier_pipeline_logistic():
    pipe = build_classifier_pipeline("logistic_regression", seed=0)
    assert "scaler" in pipe.named_steps
    assert "clf" in pipe.named_steps


def test_build_mlp_pipeline_class_weight_via_oversampling():
    pipe = build_classifier_pipeline("mlp", seed=0, params={"max_iter": 50, "hidden_layer_sizes": (8,)})
    assert getattr(pipe, "_mlp_class_weight", None) == "balanced"
    X = np.random.RandomState(0).randn(24, 6)
    y = np.array([0, 0, 0, 0, 1, 1, 2, 3] * 3, dtype=np.int64)
    X_fit, y_fit = _resample_train_for_class_weight(X, y, "balanced", seed=0)
    assert len(y_fit) == 4 * 12  # 4 classes × max count (12)
    assert len(np.unique(y_fit)) == 4
    _fit_pipeline(pipe, X, y, seed=0)
    assert hasattr(pipe.named_steps["clf"], "classes_")


def test_build_predictions_dataframe_columns():
    meta = pd.DataFrame(
        {
            "sentence": ["s1", "s2"],
            "pred_label": ["A0", "B"],
            "accident_id": [1, 2],
        }
    )
    probs = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.2]])
    preds = build_predictions_dataframe(
        meta,
        ["A0", "B"],
        probs,
        [0.7, 0.6],
        [0.6, 0.5],
        [0.5, 0.4],
        macros=MACRO_NAMES,
        method_name="test",
        text_col="sentence",
        group_col="accident_id",
        label_col="pred_label",
    )
    assert "prob_A0" in preds.columns
    assert "true_macro" in preds.columns
    assert len(preds) == 2


def test_run_model_group_kfold_cv_tiny_synthetic():
    rng = np.random.RandomState(0)
    n = 40
    X = rng.randn(n, 8)
    macros = list(MACRO_NAMES)
    y = np.array([macros[i % 4] for i in range(n)], dtype=object)
    groups = np.array([f"g{i // 4}" for i in range(n)], dtype=object)
    rows = run_model_group_kfold_cv(
        "logistic_regression",
        X,
        y,
        groups,
        macros=macros,
        n_folds=2,
        seed=0,
        params={"max_iter": 200},
    )
    assert len(rows) == 2
    assert "macro_f1" in rows[0]


def test_merge_model_registry_overrides():
    reg = merge_model_registry({"logistic_regression": {"params": {"C": 0.5}}})
    assert reg["logistic_regression"]["params"]["C"] == 0.5
    assert reg["logistic_regression"]["params"]["max_iter"] == 2000


def test_test_corpus_merge_requires_doc_id(tmp_path):
    """CSV test sans doc_id : create_doc_id_if_missing avant merge."""
    from scgm_text.dataset_text_embeddings import merge_metadata_with_embeddings
    from scgm_text.utils_io import create_doc_id_if_missing

    meta_csv = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "accident_id": ["g1", "g2"],
            "sentence": ["a", "b"],
            "pred_label": ["A0", "B"],
        }
    ).to_csv(meta_csv, index=False)
    emb_csv = tmp_path / "emb.csv"
    pd.DataFrame(
        {
            "doc_id": [1, 2],
            "dim_0": [1.0, 0.0],
            "dim_1": [0.0, 1.0],
        }
    ).to_csv(emb_csv, index=False)

    meta = pd.read_csv(meta_csv)
    slim = meta.drop(columns=[c for c in meta.columns if c.startswith("dim_")], errors="ignore")
    with pytest.raises(KeyError):
        merge_metadata_with_embeddings(slim, str(emb_csv))

    meta = create_doc_id_if_missing(meta)
    slim = meta.drop(columns=[c for c in meta.columns if c.startswith("dim_")], errors="ignore")
    merged, dim_cols = merge_metadata_with_embeddings(slim, str(emb_csv))
    assert len(merged) == 2
    assert dim_cols == ["dim_0", "dim_1"]
