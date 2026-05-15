"""Sinkhorn-Knopp E-step with diagnostics."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from sinkhornknopp import optimize_l_sk


def sinkhorn_assign(score_matrix: np.ndarray, lmd: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Assign latent components via Sinkhorn-Knopp.

    Parameters
    ----------
    score_matrix : (n, K) matrix passed to Sinkhorn (NOT macro marginal p(y|x)).
    """
    prob, argmax_q = optimize_l_sk(np.asarray(score_matrix, dtype=np.float64), lmd)
    prob = np.asarray(prob, dtype=np.float64)
    row_sums = prob.sum(axis=1, keepdims=True)
    row_sums = np.clip(row_sums, 1e-12, None)
    prob_norm = prob / row_sums
    entropy = -np.sum(prob_norm * np.log(np.clip(prob_norm, 1e-12, None)))
    active = int(np.unique(argmax_q).size)
    diagnostics = {
        "sinkhorn_assignment_entropy": float(entropy / max(prob.shape[0], 1)),
        "sinkhorn_n_active_z": float(active),
        "sinkhorn_mean_row_mass": float(prob.sum(axis=1).mean()),
    }
    return prob, argmax_q, diagnostics
