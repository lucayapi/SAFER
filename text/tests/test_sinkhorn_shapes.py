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
