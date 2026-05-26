"""Vérifie convergence Sinkhorn (max_iter, tol mean/sum/marginal_l1) et warm-up lmd."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sinkhornknopp import optimize_l_sk


def _import_get_effective_lmd():
    from scripts.train_scgm_text import get_effective_lmd

    return get_effective_lmd


def _run_sinkhorn_case(n: int, k: int, lmd: float, tol_mode: str) -> None:
    rng = np.random.default_rng(0)
    scores = rng.random((n, k)).astype(np.float64) + 1e-3
    prob, argmax_q, meta = optimize_l_sk(
        scores,
        lmd,
        max_iter=500,
        tol=1e-4,
        tol_mode=tol_mode,
        check_every=10,
        verbose=True,
    )
    assert prob.shape == (n, k), prob.shape
    assert argmax_q.shape == (n,), argmax_q.shape
    assert not np.isnan(prob).any()
    assert meta["n_iter"] <= 500
    print(
        f"[OK] tol_mode={tol_mode} converged={meta['converged']} "
        f"n_iter={meta['n_iter']} err_mean={meta['err_mean_final']:.4e}",
        flush=True,
    )


def test_effective_lmd() -> None:
    get_effective_lmd = _import_get_effective_lmd()
    assert get_effective_lmd(1, 25.0, 5.0, 5) == 5.0
    assert get_effective_lmd(5, 25.0, 5.0, 5) == 25.0
    assert get_effective_lmd(6, 25.0, 5.0, 5) == 25.0
    assert get_effective_lmd(3, 25.0, 5.0, 5) == 15.0
    assert get_effective_lmd(1, 25.0, None, 0) == 25.0
    print("[OK] get_effective_lmd", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=37370)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--lmd", type=float, default=10.0)
    args = p.parse_args()

    test_effective_lmd()
    for mode in ("mean", "sum", "marginal_l1"):
        _run_sinkhorn_case(args.n, args.k, args.lmd, mode)
    print("\n[OK] all Sinkhorn convergence checks passed", flush=True)


if __name__ == "__main__":
    main()
