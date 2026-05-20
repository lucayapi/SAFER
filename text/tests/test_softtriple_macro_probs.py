"""Tests probabilités macro SoftTriple."""

from __future__ import annotations

import numpy as np
import pytest

from macro_transfer.softtriple_macro import macro_probs_softtriple


def test_macro_probs_sum_to_one():
    rng = np.random.default_rng(0)
    n, d, c, k = 20, 16, 4, 3
    z = rng.standard_normal((n, d))
    centers = rng.standard_normal((c, k, d))
    prob_y, gamma = macro_probs_softtriple(z, centers, gamma=0.2, temperature=1.0)
    assert prob_y.shape == (n, c)
    assert np.allclose(prob_y.sum(axis=1), 1.0, atol=1e-5)
    assert gamma.shape == (n, c, k)
    assert np.allclose(gamma.sum(axis=2), 1.0, atol=1e-5)


def test_macro_probs_temperature():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((10, 8))
    centers = rng.standard_normal((4, 2, 8))
    p_cold, _ = macro_probs_softtriple(z, centers, temperature=0.5)
    p_hot, _ = macro_probs_softtriple(z, centers, temperature=5.0)
    assert p_cold.max(axis=1).mean() >= p_hot.max(axis=1).mean() - 0.05
