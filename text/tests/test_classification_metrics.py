"""Tests safer_core.classification_metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from safer_core.classification_metrics import build_gating_from_predictions, evaluate_macro_predictions


def test_evaluate_macro_predictions_available_labels():
    y_true = np.array(["A0", "A1", "A0", "A1"], dtype=object)
    y_pred = np.array(["A0", "A1", "A1", "A1"], dtype=object)
    probs = np.array(
        [
            [0.9, 0.1, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0],
            [0.4, 0.6, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    metrics = evaluate_macro_predictions(y_true, y_pred, probs, macros=["A0", "A1"])
    assert metrics["n_eval"] == 4
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert np.asarray(metrics["confusion_matrix"]).shape == (2, 2)


def test_evaluate_macro_predictions_without_mapped_labels():
    y_true = np.array(["UNK", "UNK"], dtype=object)
    y_pred = np.array(["A0", "A1"], dtype=object)
    probs = np.array([[0.8, 0.2, 0.0, 0.0], [0.3, 0.7, 0.0, 0.0]], dtype=np.float64)
    metrics = evaluate_macro_predictions(y_true, y_pred, probs, macros=["A0", "A1"])
    assert metrics["n_eval"] == 0
    assert np.isnan(metrics["balanced_accuracy"])
    assert np.isnan(metrics["macro_f1"])


def test_build_gating_from_predictions_schema():
    preds = pd.DataFrame(
        {
            "pred_macro": ["A0", "C"],
            "confidence": [0.8, 0.7],
            "prob_A0": [0.8, 0.1],
            "prob_A1": [0.1, 0.1],
            "prob_B": [0.05, 0.1],
            "prob_C": [0.05, 0.7],
        }
    )
    g = build_gating_from_predictions(preds, ["A0", "A1", "B", "C"])
    assert list(g["m_hat"]) == ["A0", "C"]
    assert "q_conf" in g.columns
    assert "ambiguous" in g.columns
    assert bool(g["ambiguous"].sum()) is False
