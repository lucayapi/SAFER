"""Tests unitaires TPN prototypes (CPU, pas de GPU)."""

from __future__ import annotations

import numpy as np
import pytest

from macro_transfer.tpn_prototypes import (
    compute_source_prototypes,
    compute_source_target_prototypes,
    compute_target_prototypes_soft,
    l2_normalize_np,
    soft_assignments,
    symmetric_kl,
)


def test_compute_source_prototypes_is_class_mean():
    h = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    labels = ["A0", "A0", "A1", "A1"]
    protos = compute_source_prototypes(h, labels, macros=("A0", "A1"))
    expected_a0 = l2_normalize_np(np.array([1.0, 0.0]))
    expected_a1 = l2_normalize_np(np.array([0.0, 1.0]))
    np.testing.assert_allclose(protos[0], expected_a0, atol=1e-6)
    np.testing.assert_allclose(protos[1], expected_a1, atol=1e-6)


def test_compute_target_prototypes_soft_one_hot():
    h = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    protos = compute_target_prototypes_soft(h, q)
    np.testing.assert_allclose(protos[0], l2_normalize_np(h[0]), atol=1e-6)
    np.testing.assert_allclose(protos[1], l2_normalize_np(h[1]), atol=1e-6)


def test_compute_target_prototypes_soft_weighted():
    h = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    q = np.array([[0.75, 0.25], [0.25, 0.75]], dtype=np.float64)
    protos = compute_target_prototypes_soft(h, q)
    expected = l2_normalize_np(0.75 * h[0] + 0.25 * h[1])
    np.testing.assert_allclose(protos[0], expected, atol=1e-5)


def test_soft_assignments_sum_to_one():
    scores = np.array([[0.0, 1.0, 2.0], [3.0, 1.0, 0.0]], dtype=np.float64)
    q = soft_assignments(scores, assignment_mode="soft")
    np.testing.assert_allclose(q.sum(axis=1), np.ones(2), atol=1e-6)


def test_compute_source_target_prototypes_rho_one():
    h_s = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    labels = ["A0", "A1"]
    h_t = np.array([[1.0, 0.0]], dtype=np.float64)
    q = np.array([[1.0, 0.0]], dtype=np.float64)
    protos = compute_source_target_prototypes(
        h_s, labels, h_t, q, rho=1.0, macros=("A0", "A1")
    )
    # A0: (h_s[0] + h_t[0]) / (1 + 1) = [0.5, 0]
    expected_a0 = l2_normalize_np(np.array([0.5, 0.0]))
    np.testing.assert_allclose(protos[0], expected_a0, atol=1e-6)


def test_symmetric_kl_identity_zero():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert symmetric_kl(p, p) == pytest.approx(0.0, abs=1e-8)
