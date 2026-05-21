"""Gating macro TPN (confiance, marge, entropie)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from macro_transfer.constants import MACRO_NAMES

EPS = 1e-8
AmbiguousRule = Literal["confidence_or_margin", "confidence_and_margin", "confidence_only"]


def build_gating_frame(
    prob_matrix: np.ndarray,
    *,
    confidence_threshold: float = 0.35,
    margin_threshold: float = 0.03,
    ambiguous_rule: AmbiguousRule = "confidence_or_margin",
) -> pd.DataFrame:
    """
    prob_matrix : (N, 4) ordre A0, A1, B, C.
    Retourne p_*, m_hat, q_conf, margin, entropy, ambiguous.
    """
    p = np.asarray(prob_matrix, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != len(MACRO_NAMES):
        raise ValueError(f"prob_matrix attendu (N, {len(MACRO_NAMES)}), reçu {p.shape}")
    row_sum = p.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 1e-12, row_sum, 1.0)
    p = p / row_sum

    idx = np.argmax(p, axis=1)
    sorted_p = np.sort(p, axis=1)
    q_conf = sorted_p[:, -1]
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    entropy = -np.sum(p * np.log(p + EPS), axis=1)

    out = {f"p_{m}": p[:, i] for i, m in enumerate(MACRO_NAMES)}
    out["m_hat"] = [MACRO_NAMES[i] for i in idx]
    out["q_conf"] = q_conf
    out["margin"] = margin
    out["entropy"] = entropy

    low_conf = q_conf < float(confidence_threshold)
    low_margin = margin < float(margin_threshold)
    if ambiguous_rule == "confidence_only":
        ambiguous = low_conf
    elif ambiguous_rule == "confidence_and_margin":
        ambiguous = low_conf & low_margin
    else:
        ambiguous = low_conf | low_margin
    out["ambiguous"] = ambiguous
    return pd.DataFrame(out)


def summarize_gating_stats(gating: pd.DataFrame) -> dict:
    n = len(gating)
    if n == 0:
        return {"n_total": 0, "n_non_ambiguous": 0, "n_ambiguous": 0, "per_macro": {}}
    amb = gating["ambiguous"].astype(bool)
    per_macro: dict = {}
    for macro in MACRO_NAMES:
        m_mask = gating["m_hat"].astype(str) == macro
        per_macro[macro] = {
            "n_m_hat": int(m_mask.sum()),
            "n_non_ambiguous": int((m_mask & ~amb).sum()),
            "n_ambiguous_only": int((m_mask & amb).sum()),
        }
    return {
        "n_total": n,
        "n_non_ambiguous": int((~amb).sum()),
        "n_ambiguous": int(amb.sum()),
        "mean_q_conf": float(gating["q_conf"].astype(float).mean()),
        "mean_margin": float(gating["margin"].astype(float).mean()),
        "mean_entropy": float(gating["entropy"].astype(float).mean()),
        "per_macro": per_macro,
    }
