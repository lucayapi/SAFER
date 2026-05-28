from __future__ import annotations

import numpy as np
import pytest

from macro_transfer.frozen_source_prototypes import (
    assign_macros_from_source_prototypes,
    compute_source_prototypes,
    evaluate_macro_predictions,
    l2_normalize,
    pairwise_distances_to_prototypes,
    softmax_over_negative_distances,
)


def test_l2_normalize_rows():
    x = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float64)
    z = l2_normalize(x)
    norms = np.linalg.norm(z, axis=1)
    assert np.allclose(norms, np.ones_like(norms), atol=1e-8)


def test_compute_source_prototypes_and_missing_macro():
    emb = np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float64)
    labels = np.array(["A0", "A0", "A1"], dtype=object)
    protos, df = compute_source_prototypes(emb, labels, macros=["A0", "A1"], normalize=True)
    assert protos.shape == (2, 2)
    assert set(df["macro"]) == {"A0", "A1"}
    with pytest.raises(ValueError, match="Aucun exemple source"):
        compute_source_prototypes(emb, labels, macros=["A0", "A1", "B"], normalize=True)


def test_pairwise_distances_cosine_and_sqeuclidean():
    z = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    p = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    d_cos = pairwise_distances_to_prototypes(z, p, metric="cosine")
    d_sq = pairwise_distances_to_prototypes(z, p, metric="sqeuclidean")
    assert d_cos.shape == (2, 2)
    assert d_sq.shape == (2, 2)
    assert d_cos[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert d_sq[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_softmax_over_negative_distances_stable():
    d = np.array([[0.0, 1.0, 2.0], [3.0, 1.0, 0.0]], dtype=np.float64)
    p = softmax_over_negative_distances(d, tau=0.5)
    assert p.shape == d.shape
    assert np.allclose(p.sum(axis=1), np.ones(2), atol=1e-8)


def test_assign_outputs_pred_conf_margin_entropy():
    z_t = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    p = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    out = assign_macros_from_source_prototypes(
        z_t,
        p,
        macros=["A0", "A1"],
        tau=0.1,
        metric="sqeuclidean",
    )
    assert list(out["pred_macro"]) == ["A0", "A1"]
    assert out["probs"].shape == (2, 2)
    assert out["distances"].shape == (2, 2)
    assert np.all(out["confidence"] >= 0.5)


def test_evaluate_macro_predictions_available_labels():
    y_true = np.array(["A0", "A1", "A0", "A1"], dtype=object)
    y_pred = np.array(["A0", "A1", "A1", "A1"], dtype=object)
    probs = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.4, 0.6],
            [0.2, 0.8],
        ],
        dtype=np.float64,
    )
    metrics = evaluate_macro_predictions(y_true, y_pred, probs, macros=["A0", "A1"])
    assert metrics["n_eval"] == 4
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "classification_report" in metrics
    assert np.asarray(metrics["confusion_matrix"]).shape == (2, 2)


def test_evaluate_macro_predictions_without_mapped_labels():
    y_true = np.array(["UNK", "UNK"], dtype=object)
    y_pred = np.array(["A0", "A1"], dtype=object)
    probs = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
    metrics = evaluate_macro_predictions(y_true, y_pred, probs, macros=["A0", "A1"])
    assert metrics["n_eval"] == 0
    assert np.isnan(metrics["balanced_accuracy"])
    assert np.isnan(metrics["macro_f1"])
    assert np.asarray(metrics["confusion_matrix"]).shape == (2, 2)

