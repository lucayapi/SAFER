"""Tests unitaires prototypes macro (CPU, pas de GPU)."""

from __future__ import annotations

import numpy as np
import pytest

from macro_transfer.prototypes import (
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
    assert protos.shape == (2, 2)
    assert np.allclose(protos[0], l2_normalize_np(np.array([1.0, 0.0])))
    assert np.allclose(protos[1], l2_normalize_np(np.array([0.0, 1.0])))


def test_soft_assignments_sum_to_one():
    scores = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float64)
    q = soft_assignments(scores, assignment_mode="soft")
    assert np.allclose(q.sum(axis=1), 1.0)


def test_symmetric_kl_zero_for_identical():
    p = np.array([0.5, 0.5])
    assert symmetric_kl(p, p) == pytest.approx(0.0, abs=1e-6)


def test_compute_target_prototypes_soft_weighted():
    h = np.eye(2, dtype=np.float64)
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    protos = compute_target_prototypes_soft(h, q)
    assert protos.shape == (2, 2)


def test_compute_source_target_prototypes_mix():
    h_s = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    labels = ["A0", "A1"]
    h_t = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
    q = np.array([[0.8, 0.2], [0.2, 0.8]])
    protos = compute_source_target_prototypes(h_s, labels, h_t, q, rho=1.0, macros=("A0", "A1"))
    assert protos.shape == (2, 2)
    norms = np.linalg.norm(protos, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
