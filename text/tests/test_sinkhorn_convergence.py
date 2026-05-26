"""Tests convergence Sinkhorn (max_iter, tol modes, lmd warm-up)."""

import numpy as np
import pytest

from sinkhornknopp import optimize_l_sk
from scgm_text.sinkhorn_estep import sinkhorn_assign


def test_optimize_l_sk_converges_mean_small_n():
    rng = np.random.default_rng(0)
    n, k = 500, 8
    scores = rng.random((n, k)).astype(np.float64) + 1e-3
    prob, argmax_q, meta = optimize_l_sk(
        scores, lmd=10.0, max_iter=500, tol=1e-4, tol_mode="mean", verbose=False
    )
    assert prob.shape == (n, k)
    assert len(argmax_q) == n
    assert not np.isnan(prob).any()
    assert meta["converged"] is True
    assert meta["n_iter"] <= 500


def test_optimize_l_sk_max_iter_returns_without_convergence():
    rng = np.random.default_rng(1)
    scores = rng.random((200, 6)).astype(np.float64) + 1e-3
    _, argmax_q, meta = optimize_l_sk(
        scores,
        lmd=25.0,
        max_iter=5,
        tol=1e-12,
        tol_mode="mean",
        verbose=False,
    )
    assert len(argmax_q) == 200
    assert meta["converged"] is False
    assert meta["n_iter"] == 5


def test_sinkhorn_assign_exposes_convergence_diag():
    rng = np.random.default_rng(2)
    scores = rng.random((50, 4)).astype(np.float64) + 1e-3
    _, _, diag = sinkhorn_assign(scores, lmd=10.0, max_iter=200, verbose=False)
    assert "sinkhorn_converged" in diag
    assert "sinkhorn_n_iter" in diag
    assert "sinkhorn_err_mean_final" in diag


@pytest.mark.parametrize("tol_mode", ["mean", "sum", "marginal_l1"])
def test_tol_modes_smoke(tol_mode: str):
    rng = np.random.default_rng(3)
    scores = rng.random((300, 6)).astype(np.float64) + 1e-2
    _, _, meta = optimize_l_sk(
        scores,
        lmd=8.0,
        max_iter=300,
        tol=1e-3,
        tol_mode=tol_mode,
        verbose=False,
    )
    assert meta["n_iter"] <= 300


def test_get_effective_lmd_warmup():
    from scripts.train_scgm_text import get_effective_lmd

    assert get_effective_lmd(1, 25.0, 5.0, 5) == pytest.approx(5.0)
    assert get_effective_lmd(5, 25.0, 5.0, 5) == pytest.approx(25.0)
    assert get_effective_lmd(6, 25.0, 5.0, 5) == pytest.approx(25.0)
    assert get_effective_lmd(3, 25.0, 5.0, 5) == pytest.approx(15.0)
    assert get_effective_lmd(1, 25.0, None, 0) == pytest.approx(25.0)
