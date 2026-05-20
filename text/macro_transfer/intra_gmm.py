"""GMM intra-macro sur embeddings projetés."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from macro_transfer.constants import MACRO_NAMES
from macro_transfer.topics_export import export_themes_by_macro


def _select_n_components(
    z: np.ndarray,
    *,
    k_min: int = 2,
    k_max: int = 8,
    covariance_type: str = "full",
    random_state: int = 42,
) -> int:
    if z.shape[0] < k_min:
        return max(1, z.shape[0])
    best_k = k_min
    best_bic = np.inf
    for k in range(k_min, min(k_max, z.shape[0]) + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=random_state,
        )
        gmm.fit(z)
        bic = gmm.bic(z)
        if bic < best_bic:
            best_bic = bic
            best_k = k
    return best_k


def fit_gmm_per_macro(
    z: np.ndarray,
    meta: pd.DataFrame,
    gating: pd.DataFrame,
    *,
    method: str = "gmm",
    confidence_threshold: float = 0.5,
    n_components: Optional[Dict[str, int]] = None,
    k_min: int = 2,
    k_max: int = 8,
    covariance_type: str = "full",
    max_iter: int = 200,
    n_init: int = 1,
    reg_covar: float = 1.0e-6,
    random_state: int = 42,
    output_dir: Path,
    sentence_col: str = "sentence",
    top_k_words: int = 12,
    top_k_sentences: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Par macro (non ambigu) : GMM → gamma_jmk, p_mk = p_m * gamma.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prob_cols = [f"p_{m}" for m in MACRO_NAMES]
    p_macro = gating[prob_cols].to_numpy(dtype=np.float64)

    assign_rows: list[dict] = []
    theme_parts: list[pd.DataFrame] = []

    for macro in MACRO_NAMES:
        mi = MACRO_NAMES.index(macro)
        mask = (
            (gating["m_hat"].astype(str) == macro)
            & (~gating["ambiguous"].astype(bool))
        )
        if not mask.any():
            continue
        idx = np.where(mask.to_numpy())[0]
        z_sub = z[idx]
        if len(idx) < 2:
            for j, doc_idx in enumerate(idx):
                assign_rows.append(
                    {
                        "doc_idx": int(doc_idx),
                        "macro": macro,
                        "topic_id": 0,
                        "gamma": 1.0,
                        "p_mk": float(p_macro[doc_idx, mi]),
                    }
                )
            continue

        if n_components and macro in n_components:
            k = int(n_components[macro])
        else:
            k = _select_n_components(
                z_sub, k_min=k_min, k_max=k_max, covariance_type=covariance_type, random_state=random_state
            )
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            max_iter=max_iter,
            n_init=n_init,
            reg_covar=reg_covar,
            random_state=random_state,
        )
        gmm.fit(z_sub)
        gamma = gmm.predict_proba(z_sub)
        topic_ids = gamma.argmax(axis=1)

        for local_j, doc_idx in enumerate(idx):
            tid = int(topic_ids[local_j])
            g = float(gamma[local_j, tid])
            assign_rows.append(
                {
                    "doc_idx": int(doc_idx),
                    "macro": macro,
                    "topic_id": tid,
                    "gamma": g,
                    "p_mk": float(p_macro[doc_idx, mi] * g),
                }
            )

        sub_assign = pd.DataFrame(
            [
                {
                    "doc_idx": int(doc_idx),
                    "m_hat": macro,
                    "topic_id": int(topic_ids[local_j]),
                }
                for local_j, doc_idx in enumerate(idx)
            ]
        )
        themes = export_themes_by_macro(
            sub_assign,
            meta,
            z,
            method=method,
            topic_col="topic_id",
            macro_col="m_hat",
            output_path=output_dir / f"themes_macro_{macro}.csv",
            sentence_col=sentence_col,
            top_k_words=top_k_words,
            top_k_sentences=top_k_sentences,
        )
        theme_parts.append(themes)

    assignments = pd.DataFrame(assign_rows)
    assignments.to_csv(output_dir / "assignments.csv", index=False)
    themes_all = pd.concat(theme_parts, ignore_index=True) if theme_parts else pd.DataFrame()
    if len(themes_all):
        themes_all.to_csv(output_dir / "themes_by_macro.csv", index=False)
    return themes_all, assignments
