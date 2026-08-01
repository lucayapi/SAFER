"""Tests rééquilibrage supervised_macro_ft."""

from __future__ import annotations

import numpy as np
import pytest

from supervised_macro_ft.class_balance import (
    balanced_oversample_arrays,
    balanced_oversample_indices,
    resolve_train_balance,
)


def test_balanced_oversample_indices_balanced_classes():
    y = np.array([0, 0, 0, 0, 1, 1, 2, 3] * 3, dtype=np.int64)
    idx = balanced_oversample_indices(y, seed=0)
    assert len(idx) == 4 * 12
    sub_y = y[idx]
    for cls in np.unique(y):
        assert np.sum(sub_y == cls) == 12


def test_balanced_oversample_indices_reproducible():
    y = np.array([0, 0, 1, 2], dtype=np.int64)
    a = balanced_oversample_indices(y, seed=7)
    b = balanced_oversample_indices(y, seed=7)
    np.testing.assert_array_equal(a, b)


def test_balanced_oversample_arrays_matches_indices():
    X = np.random.RandomState(0).randn(24, 4)
    y = np.array([0, 0, 0, 0, 1, 1, 2, 3] * 3, dtype=np.int64)
    X_fit, y_fit = balanced_oversample_arrays(X, y, seed=0)
    assert len(y_fit) == 4 * 12
    assert X_fit.shape[0] == len(y_fit)


def test_resolve_train_balance_oversampling_rejected():
    with pytest.raises(ValueError, match="oversampling n'est plus supporté"):
        resolve_train_balance({"oversampling": True, "class_weight": "balanced"})
    with pytest.raises(ValueError, match="oversampling n'est plus supporté"):
        resolve_train_balance({"oversampling": True})


def test_resolve_train_balance_defaults():
    use_os, cw = resolve_train_balance({})
    assert use_os is False
    assert cw is None


def test_resolve_train_balance_class_weight_only():
    use_os, cw = resolve_train_balance({"class_weight": "balanced"})
    assert use_os is False
    assert cw == "balanced"


def test_resolve_train_balance_rejects_unknown_class_weight():
    with pytest.raises(ValueError, match="class_weight non supporté"):
        resolve_train_balance({"class_weight": "inverse"})
