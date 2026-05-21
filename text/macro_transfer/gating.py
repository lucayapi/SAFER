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


def summarize_gating_stats(gating: pd.DataFrame) -> dict:
    """Comptages pour diagnostic BERTopic (par macro, ambiguës vs confiantes)."""
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
        "per_macro": per_macro,
    }


def format_gating_stats_message(stats: dict, *, confidence_threshold: float) -> str:
    """Message lisible pour logs / erreurs."""
    lines = [
        f"  unités totales : {stats['n_total']}",
        f"  non ambiguës (q_conf >= {confidence_threshold}) : {stats['n_non_ambiguous']}",
        f"  ambiguës : {stats['n_ambiguous']}",
        f"  q_conf moyen : {stats.get('mean_q_conf', float('nan')):.4f}",
        "  par macro (m_hat, non ambiguës) :",
    ]
    for macro, counts in (stats.get("per_macro") or {}).items():
        lines.append(
            f"    {macro}: m_hat={counts['n_m_hat']}, "
            f"non_ambiguës={counts['n_non_ambiguous']}"
        )
    return "\n".join(lines)
