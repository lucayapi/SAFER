"""Export tables de topics intra-macro (mots / phrases)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from macro_transfer.constants import MACRO_NAMES
from scgm_text.topic_export import _top_sentences_by_distance, _top_words_for_texts


def build_topic_row(
    method: str,
    macro: str,
    topic_id: int,
    subset: pd.DataFrame,
    z_subset: np.ndarray,
    *,
    sentence_col: str = "sentence",
    centroid: Optional[np.ndarray] = None,
    top_k_words: int = 12,
    top_k_sentences: int = 5,
) -> dict:
    sentences = subset[sentence_col].astype(str).tolist()
    top_words = _top_words_for_texts(sentences, top_k=top_k_words)
    if centroid is not None and len(sentences) > 0:
        top_sentences = _top_sentences_by_distance(
            sentences, z_subset, centroid, top_k=top_k_sentences
        )
    else:
        top_sentences = " || ".join(sentences[:top_k_sentences])
    return {
        "method": method,
        "macro": macro,
        "topic_id": int(topic_id),
        "n_units": int(len(subset)),
        "top_words": top_words,
        "top_sentences": top_sentences,
    }


def export_themes_by_macro(
    assignments: pd.DataFrame,
    meta: pd.DataFrame,
    z: np.ndarray,
    *,
    method: str,
    topic_col: str,
    macro_col: str = "m_hat",
    output_path: Path,
    sentence_col: str = "sentence",
    top_k_words: int = 12,
    top_k_sentences: int = 5,
) -> pd.DataFrame:
    """Agrège themes_by_macro.csv depuis assignations unitaires (``doc_idx`` global)."""
    rows = []
    if "doc_idx" not in assignments.columns:
        raise ValueError("assignments doit contenir doc_idx")

    for macro in MACRO_NAMES:
        m_mask = assignments[macro_col].astype(str) == macro
        if not m_mask.any():
            continue
        sub_m = assignments.loc[m_mask]
        for topic_id in sorted(sub_m[topic_col].dropna().unique()):
            sub_t = sub_m.loc[sub_m[topic_col] == topic_id]
            idx = sub_t["doc_idx"].to_numpy(dtype=np.int64)
            subset = meta.iloc[idx]
            z_sub = z[idx]
            centroid = z_sub.mean(axis=0) if len(z_sub) else None
            row = build_topic_row(
                method,
                macro,
                int(topic_id),
                subset,
                z_sub,
                sentence_col=sentence_col,
                centroid=centroid,
                top_k_words=top_k_words,
                top_k_sentences=top_k_sentences,
            )
            rows.append(row)

    themes = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    themes.to_csv(output_path, index=False)
    return themes


def comparison_bertopic_vs_gmm(
    bertopic_themes: pd.DataFrame,
    gmm_themes: pd.DataFrame,
) -> pd.DataFrame:
    """Effectifs par macro / méthode pour summary."""
    rows = []
    for method, df in (("bertopic", bertopic_themes), ("gmm", gmm_themes)):
        if df.empty:
            continue
        for macro in MACRO_NAMES:
            sub = df[df["macro"].astype(str) == macro]
            rows.append(
                {
                    "method": method,
                    "macro": macro,
                    "n_topics": int(sub["topic_id"].nunique()) if len(sub) else 0,
                    "n_units": int(sub["n_units"].sum()) if "n_units" in sub.columns else 0,
                    "empty_topics": int((sub["n_units"] == 0).sum()) if len(sub) else 0,
                }
            )
    return pd.DataFrame(rows)
