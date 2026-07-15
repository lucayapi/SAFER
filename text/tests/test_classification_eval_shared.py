"""Tests safer_core.classification_eval."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from safer_core.classification_eval import (
    build_and_save_predictions,
    build_cv_summary_from_kfold,
    evaluate_classifier_on_embeddings,
    export_projected_embeddings,
    fit_logistic_and_evaluate,
    fit_logistic_on_embeddings,
    load_saved_predictions,
    resolve_test_corpora,
    save_classification_outputs,
    save_corpus_predictions,
    summarize_ood_classification,
)


def test_resolve_test_corpora():
    assert resolve_test_corpora({"test_corpora": ["metallurgie", "caou"]}) == [
        "metallurgie",
        "caou",
    ]


def test_summarize_ood_classification():
    cv = pd.DataFrame([{"mean_balanced_accuracy": 0.8, "std_balanced_accuracy": 0.05}])
    summary = summarize_ood_classification(
        {
            "metallurgie": {"balanced_accuracy": 0.6, "macro_f1": 0.5, "accuracy": 0.55},
            "caou": {"balanced_accuracy": 0.4, "macro_f1": 0.35, "accuracy": 0.38},
        },
        cv,
        model_name="batch_triplet",
    )
    assert summary.loc[0, "ba_ood_avg"] == pytest.approx(0.5)
    assert summary.loc[0, "ba_ood_worst"] == pytest.approx(0.4)


def test_fit_logistic_and_evaluate_toy():
    rng = np.random.RandomState(0)
    X_train = rng.randn(40, 8)
    y_train = np.array([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10)
    X_eval = rng.randn(20, 8)
    y_eval = np.array(["A0"] * 5 + ["A1"] * 5 + ["B"] * 5 + ["C"] * 5)
    metrics = fit_logistic_and_evaluate(X_train, y_train, X_eval, y_eval, seed=0)
    assert "balanced_accuracy" in metrics


def test_export_projected_embeddings(tmp_path: Path):
    emb = np.ones((3, 4))
    meta = pd.DataFrame(
        {
            "doc_id": ["a", "b", "c"],
            "pred_label": ["A0", "A1", "B"],
            "accident_id": [1, 2, 3],
        }
    )
    npy, csv = export_projected_embeddings(emb, meta, tmp_path, "btp", label_col="pred_label")
    assert npy.is_file()
    assert csv.is_file()
    assert np.load(npy).shape == (3, 4)


def test_save_classification_outputs(tmp_path: Path):
    cv = pd.DataFrame([{"mean_val_balanced_accuracy": 0.7, "std_val_balanced_accuracy": 0.1}])
    paths = save_classification_outputs(
        tmp_path,
        method_name="supcon",
        metrics_by_corpus={
            "btp": {"balanced_accuracy": 0.75, "macro_f1": 0.7, "accuracy": 0.72},
            "metallurgie": {"balanced_accuracy": 0.5, "macro_f1": 0.45, "accuracy": 0.48},
            "caou": {"balanced_accuracy": 0.4, "macro_f1": 0.35, "accuracy": 0.38},
        },
        cv_summary=build_cv_summary_from_kfold(cv, model_name="supcon"),
    )
    assert paths["cross_domain"].is_file()
    assert (tmp_path / "metrics" / "metrics_classification_test_metallurgie.csv").is_file()


def test_evaluate_classifier_return_details_and_save_predictions(tmp_path: Path):
    rng = np.random.RandomState(0)
    X_train = rng.randn(40, 8)
    y_train = np.array([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10)
    for i, c in enumerate(y_train):
        X_train[i, c] += 4.0
    X_eval = X_train.copy()
    y_eval = np.array(["A0"] * 10 + ["A1"] * 10 + ["B"] * 10 + ["C"] * 10)
    meta = pd.DataFrame(
        {
            "sentence": [f"t{i}" for i in range(40)],
            "pred_label": y_eval,
            "accident_id": list(range(40)),
        }
    )
    pipe = fit_logistic_on_embeddings(X_train, y_train, seed=0)
    metrics, details = evaluate_classifier_on_embeddings(
        pipe, X_eval, y_eval, return_details=True
    )
    assert "balanced_accuracy" in metrics
    assert "pred_macro" in details
    assert details["probs"].shape == (40, 4)

    preds, path = build_and_save_predictions(
        meta,
        details,
        tmp_path,
        "btp",
        method_name="test_method",
        also_transfer_alias=True,
    )
    assert path.is_file()
    assert (tmp_path / "transfer" / "target_macro_predictions.csv").is_file()
    loaded = load_saved_predictions(tmp_path, "btp")
    assert loaded is not None
    assert len(loaded) == 40
    assert "pred_macro" in loaded.columns
    assert load_saved_predictions(tmp_path, "missing") is None


def test_save_corpus_predictions_adds_corpus_column(tmp_path: Path):
    df = pd.DataFrame(
        {
            "pred_macro": ["A0", "A1"],
            "confidence": [0.9, 0.8],
            "sentence": ["a", "b"],
        }
    )
    path = save_corpus_predictions(df, tmp_path, "metallurgie")
    assert path.name == "predictions_metallurgie.csv"
    out = pd.read_csv(path)
    assert list(out["corpus"]) == ["metallurgie", "metallurgie"]
