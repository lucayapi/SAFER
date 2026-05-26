import numpy as np

from scgm_text.sinkhorn_estep import sinkhorn_assign


def test_sinkhorn_assign_shapes():
    rng = np.random.default_rng(0)
    scores = rng.random((20, 6)).astype(np.float64) + 1e-3
    prob, argmax_q, diag = sinkhorn_assign(scores, lmd=25.0)
    assert prob.shape == scores.shape
    assert argmax_q.shape == (20,)
    assert 0 < diag["sinkhorn_n_active_z"] <= 6
    assert "sinkhorn_assignment_entropy" in diag
    assert "sinkhorn_converged" in diag
    assert "sinkhorn_n_iter" in diag


def test_sinkhorn_assign_macro_balanced_shapes():
    rng = np.random.default_rng(1)
    scores = rng.random((20, 6)).astype(np.float64) + 1e-3
    labels = rng.integers(0, 4, size=20)
    prob, argmax_q, diag = sinkhorn_assign(
        scores,
        lmd=25.0,
        labels=labels,
        sample_prior="macro_balanced",
    )
    assert prob.shape == scores.shape
    assert argmax_q.shape == (20,)
    assert diag["sinkhorn_sample_prior"] == "macro_balanced"
