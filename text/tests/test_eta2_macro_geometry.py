import warnings

import numpy as np

from metrics.embedding_geometry_separation import (
    build_geometry_metrics_row,
    compute_eta2_macro_geometry,
)


def _orthogonal_dirs(n_dims: int = 8) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.randn(n_dims, n_dims))
    return q


def test_separated_macros_high_eta2():
    rng = np.random.default_rng(0)
    basis = _orthogonal_dirs(8)
    rows, labels = [], []
    for mid, name in enumerate(["A0", "A1", "B", "C"]):
        for _ in range(20):
            rows.append(basis[mid] * 5.0 + rng.normal(scale=0.05, size=8))
            labels.append(name)
    z = np.asarray(rows, dtype=np.float64)
    out = compute_eta2_macro_geometry(z, np.asarray(labels))
    assert out["eta2_macro_balanced"] > 0.5
    assert out["eta2_weighted"] > 0.3
    assert abs(out["B_macro_balanced"] - (out["T_macro_balanced"] - out["W_macro_balanced"])) < 1e-9
    assert abs(out["B_weighted"] - (out["T_weighted"] - out["W_weighted"])) < 1e-9


def test_random_labels_low_eta2():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(80, 16))
    labels = rng.choice(["A0", "A1", "B", "C"], size=80)
    out = compute_eta2_macro_geometry(z, labels)
    assert out["eta2_macro_balanced"] < 0.15
    assert out["eta2_weighted"] < 0.15


def test_single_sample_macro_ignored():
    rng = np.random.default_rng(2)
    basis = _orthogonal_dirs(8)
    rows, labels = [], []
    for mid, name in enumerate(["A0", "A1", "B", "C"]):
        count = 1 if name == "B" else 5
        for _ in range(count):
            rows.append(basis[mid] + rng.normal(scale=0.01, size=8))
            labels.append(name)
    out = compute_eta2_macro_geometry(np.asarray(rows), np.asarray(labels))
    assert "B" in out["macros_ignored"].split(",")
    assert np.isnan(out["W_B"])
    assert np.isfinite(out["eta2_macro_balanced"])


def test_near_zero_total_inertia_warns():
    z = np.ones((10, 4), dtype=np.float64)
    labels = np.array(["A0"] * 5 + ["A1"] * 5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = compute_eta2_macro_geometry(z, labels)
    assert np.isnan(out["eta2_macro_balanced"])
    assert any("near zero" in str(w.message).lower() for w in caught)


def test_build_geometry_metrics_row_has_rankme():
    z = np.random.default_rng(3).normal(size=(30, 8))
    labels = ["A0"] * 10 + ["A1"] * 10 + ["B"] * 5 + ["C"] * 5
    row = build_geometry_metrics_row(z, labels, method="SCGM")
    assert row["method"] == "SCGM"
    assert "rankme_global" in row
    assert "eta2_macro_balanced" in row
    assert "delta_macro" not in row
