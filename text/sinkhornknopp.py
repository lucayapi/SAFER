"""Sinkhorn-Knopp avec marges optionnelles et critères d'arrêt configurables."""

from __future__ import annotations

from time import time
from typing import Any, Dict, Tuple

import numpy as np

SINKHORN_TOL_MODES = frozenset({"mean", "sum", "marginal_l1"})


def _normalize_tol_mode(tol_mode: str) -> str:
    mode = str(tol_mode).strip().lower()
    if mode not in SINKHORN_TOL_MODES:
        raise ValueError(
            f"sinkhorn_tol_mode must be one of: {', '.join(sorted(SINKHORN_TOL_MODES))} "
            f"(got {tol_mode!r})"
        )
    return mode


def _marginal_l1_error(
    kernel: np.ndarray,
    r: np.ndarray,
    c: np.ndarray,
    a_vec: np.ndarray,
    b_vec: np.ndarray,
) -> Tuple[float, float, float]:
    """kernel (k, n), r (k,1), c (n,1); return max(row_l1, col_l1), row_l1, col_l1."""
    q = kernel * r * c.T
    row_sum = q.sum(axis=1)
    col_sum = q.sum(axis=0)
    row_l1 = float(np.abs(row_sum - a_vec).sum())
    col_l1 = float(np.abs(col_sum - b_vec).sum())
    return max(row_l1, col_l1), row_l1, col_l1


def optimize_l_sk(
    prob,
    lmd,
    a=None,
    b=None,
    *,
    ddtype=np.float64,
    max_iter: int = 500,
    tol: float = 1e-4,
    tol_mode: str = "mean",
    check_every: int = 10,
    eps: float = 1e-12,
    normalize_input: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Sinkhorn-Knopp with optional marginals.

    prob : (n_samples, n_latents) cost/score matrix.
    a : (n_latents,) row marginal target after transpose (latent), sum=1.
    b : (n_samples,) column marginal target (sample), sum=1.
    """
    tt = time()
    mode = _normalize_tol_mode(tol_mode)
    max_iter = int(max_iter)
    check_every = max(1, int(check_every))

    n_samples = prob.shape[0]
    k = prob.shape[1]

    prob = np.asarray(prob, dtype=ddtype)
    prob = np.clip(prob, eps, None)
    if normalize_input:
        row_sums = prob.sum(axis=1, keepdims=True)
        prob = prob / np.clip(row_sums, eps, None)
    prob = prob.T  # (k, n_samples)

    if a is None:
        a_vec = np.full(k, ddtype(1.0 / k), dtype=ddtype)
    else:
        a_vec = np.asarray(a, dtype=ddtype).reshape(-1)
        if a_vec.shape[0] != k:
            raise ValueError(f"a length {a_vec.shape[0]} != n_latents {k}")

    if b is None:
        b_vec = np.full(n_samples, ddtype(1.0 / n_samples), dtype=ddtype)
    else:
        b_vec = np.asarray(b, dtype=ddtype).reshape(-1)
        if b_vec.shape[0] != n_samples:
            raise ValueError(f"b length {b_vec.shape[0]} != n_samples {n_samples}")

    a_col = a_vec.reshape(k, 1)
    b_col = b_vec.reshape(n_samples, 1)

    prob = np.clip(prob ** ddtype(lmd), eps, None)

    c = np.ones((n_samples, 1), dtype=ddtype) / n_samples
    r = np.ones((k, 1), dtype=ddtype)

    converged = False
    n_iter = 0
    err_final = err_sum_final = err_mean_final = float("inf")
    row_l1_final = col_l1_final = float("nan")
    stop_err = float("inf")

    for it in range(1, max_iter + 1):
        r = a_col / np.clip(prob @ c, eps, None)
        c_new = b_col / np.clip((r.T @ prob).T, eps, None)

        should_check = (it % check_every == 0) or (it == max_iter)
        if should_check:
            ratio = np.abs(c / np.clip(c_new, eps, None) - 1.0)
            err_sum = float(np.nansum(ratio))
            err_mean = err_sum / max(n_samples, 1)

            if mode == "mean":
                stop_err = err_mean
            elif mode == "sum":
                stop_err = err_sum
            else:
                stop_err, row_l1_final, col_l1_final = _marginal_l1_error(
                    prob, r, c_new, a_vec, b_vec
                )

            err_final = stop_err
            err_sum_final = err_sum
            err_mean_final = err_mean

            if mode != "marginal_l1":
                stop_err_marg, row_l1_final, col_l1_final = _marginal_l1_error(
                    prob, r, c_new, a_vec, b_vec
                )
            else:
                stop_err_marg = stop_err

            if verbose and (it == 1 or it % 100 == 0 or stop_err < tol):
                print(
                    f"sinkhornknopp: iter {it} err={stop_err:.4e} mode={mode} "
                    f"ratio_sum={err_sum:.4f} ratio_mean={err_mean:.4e} "
                    f"row_l1={row_l1_final:.4e} col_l1={col_l1_final:.4e} "
                    f"(n={n_samples}, k={k})",
                    flush=True,
                )

            if stop_err < tol:
                converged = True
                n_iter = it
                c = c_new
                break

        c = c_new
        n_iter = it

    if not converged and verbose:
        print(
            f"[WARN] Sinkhorn reached max_iter={max_iter} without strict convergence; "
            "using last transport plan.",
            flush=True,
        )

    prob_out = prob * np.squeeze(c)
    prob_out = prob_out.T
    prob_out = prob_out * np.squeeze(r)
    argmaxes = np.nanargmax(prob_out, axis=1)

    if verbose:
        print(
            f"opt took {(time() - tt) / 60.0:.2f}min, {n_iter:4d} iters, "
            f"converged={converged}",
            flush=True,
        )

    meta: Dict[str, Any] = {
        "converged": converged,
        "n_iter": int(n_iter),
        "err_final": float(err_final),
        "err_sum_final": float(err_sum_final),
        "err_mean_final": float(err_mean_final),
        "row_l1_final": float(row_l1_final),
        "col_l1_final": float(col_l1_final),
        "tol_mode": mode,
        "tol": float(tol),
        "max_iter": max_iter,
        "lmd": float(lmd),
    }
    return prob_out, argmaxes, meta
