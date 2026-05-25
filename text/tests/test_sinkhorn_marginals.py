"""Tests Sinkhorn sample priors uniform vs macro_balanced."""

import numpy as np
import pytest
import torch

from scgm_text.sinkhorn_estep import (
    build_sinkhorn_marginals,
    macro_masses_in_b,
    sinkhorn_assign,
)


def _imbalanced_labels(n: int = 1000) -> torch.Tensor:
    """Skewed toward class 0 (A0-like)."""
    rng = np.random.default_rng(0)
    probs = np.array([0.55, 0.12, 0.08, 0.25])
    return torch.tensor(rng.choice(4, size=n, p=probs), dtype=torch.long)


def test_uniform_macro_masses_unequal():
    labels = _imbalanced_labels()
    _, b = build_sinkhorn_marginals(labels, n_latents=32, n_classes=4, sample_prior="uniform")
    masses = macro_masses_in_b(b, labels, n_classes=4)
    assert masses[0] > masses[2]
    assert masses[0] > 0.5


def test_macro_balanced_equal_mass_per_present_class():
    labels = _imbalanced_labels()
    _, b = build_sinkhorn_marginals(
        labels, n_latents=32, n_classes=4, sample_prior="macro_balanced"
    )
    c_present = len(torch.unique(labels))
    for m in torch.unique(labels).tolist():
        mass = float(b[labels == m].sum())
        assert abs(mass - 1.0 / c_present) < 1e-5


def test_marginals_sum_to_one():
    labels = _imbalanced_labels(200)
    a, b = build_sinkhorn_marginals(labels, n_latents=16, n_classes=4, sample_prior="macro_balanced")
    assert abs(float(a.sum()) - 1.0) < 1e-5
    assert abs(float(b.sum()) - 1.0) < 1e-5


def test_invalid_sample_prior():
    labels = torch.tensor([0, 1, 2, 3])
    with pytest.raises(ValueError, match="sinkhorn_sample_prior"):
        build_sinkhorn_marginals(labels, n_latents=8, n_classes=4, sample_prior="invalid")


def test_sinkhorn_assign_with_macro_balanced_smoke():
    rng = np.random.default_rng(1)
    n, r = 20, 6
    scores = rng.random((n, r)).astype(np.float64) + 1e-3
    labels = rng.integers(0, 4, size=n)
    prob, argmax_q, diag = sinkhorn_assign(
        scores,
        lmd=25.0,
        labels=labels,
        n_classes=4,
        sample_prior="macro_balanced",
        log_marginals=False,
    )
    assert prob.shape == (n, r)
    assert argmax_q.shape == (n,)
    assert diag["sinkhorn_sample_prior"] == "macro_balanced"


def test_missing_macro_class_uses_c_present():
    labels = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    _, b = build_sinkhorn_marginals(labels, n_latents=4, n_classes=4, sample_prior="macro_balanced")
    c_present = 2
    for m in (0, 1):
        assert abs(float(b[labels == m].sum()) - 1.0 / c_present) < 1e-5
