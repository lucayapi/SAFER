"""Affectation macro dure et confiance."""

from __future__ import annotations

import numpy as np
import pandas as pd

from macro_transfer.constants import MACRO_NAMES


def macro_prob_columns() -> list[str]:
    return [f"p_{m}" for m in MACRO_NAMES]


def apply_macro_gating(
    prob_matrix: np.ndarray,
    *,
    confidence_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    ``prob_matrix`` : (N, 4) colonnes dans l'ordre A0, A1, B, C.
    Retourne colonnes p_*, m_hat, q_conf, ambiguous.
    """
    p = np.asarray(prob_matrix, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != len(MACRO_NAMES):
        raise ValueError(f"prob_matrix attendu (N, {len(MACRO_NAMES)}), reçu {p.shape}")
    row_sum = p.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 1e-12, row_sum, 1.0)
    p = p / row_sum
    idx = np.argmax(p, axis=1)
    q_conf = p.max(axis=1)
    out = {f"p_{m}": p[:, i] for i, m in enumerate(MACRO_NAMES)}
    out["m_hat"] = [MACRO_NAMES[i] for i in idx]
    out["q_conf"] = q_conf
    out["ambiguous"] = q_conf < float(confidence_threshold)
    return pd.DataFrame(out)
