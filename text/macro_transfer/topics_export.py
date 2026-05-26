"""Export tables de topics intra-macro (mots / phrases)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

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
    top_words: Optional[str] = None,
) -> dict:
    sentences = subset[sentence_col].astype(str).tolist()
    words = top_words if top_words is not None else _top_words_for_texts(sentences, top_k=top_k_words)
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
        "top_words": words,
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


def summarize_topics_by_macro(themes: pd.DataFrame) -> pd.DataFrame:
    """Effectifs par macro pour summary/topics_summary.csv."""
    rows = []
    if themes.empty:
        return pd.DataFrame(columns=["macro", "n_topics", "n_units", "empty_topics"])
    for macro in MACRO_NAMES:
        sub = themes[themes["macro"].astype(str) == macro]
        rows.append(
            {
                "macro": macro,
                "n_topics": int(sub["topic_id"].nunique()) if len(sub) else 0,
                "n_units": int(sub["n_units"].sum()) if "n_units" in sub.columns else 0,
                "empty_topics": int((sub["n_units"] == 0).sum()) if len(sub) else 0,
            }
        )
    return pd.DataFrame(rows)


def _theme_label_for(
    themes: pd.DataFrame,
    macro: str,
    topic_id: int,
) -> str:
    if themes.empty or "macro" not in themes.columns or "topic_id" not in themes.columns:
        return ""
    sub = themes.loc[
        (themes["macro"].astype(str) == str(macro))
        & (themes["topic_id"].astype(int) == int(topic_id))
    ]
    if sub.empty:
        return ""
    row = sub.iloc[0]
    for col in ("theme_label", "theme_title", "theme_summary", "top_words"):
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    return ""


def _dominant_topic_from_assignments(
    assignments: pd.DataFrame,
    macro: str,
) -> tuple[int, str]:
    """Topic_id le plus fréquent (hors bruit) pour une macro."""
    if assignments.empty:
        return -1, ""
    macro_col = "macro" if "macro" in assignments.columns else "m_hat"
    if macro_col not in assignments.columns:
        return -1, ""
    sub = assignments.loc[assignments[macro_col].astype(str) == str(macro)].copy()
    if "topic_id" not in sub.columns:
        return -1, ""
    valid = sub.loc[sub["topic_id"].astype(int) >= 0]
    if valid.empty:
        return -1, ""
    counts = valid["topic_id"].astype(int).value_counts()
    tid = int(counts.index[0])
    return tid, ""


def build_macro_topic_test_table(
    macro_topic_counts: Dict[str, Any],
    assignments: pd.DataFrame,
    themes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tableau récapitulatif topics par macro (corpus test).

    Colonnes : macro, n_units, n_topics, bruit_pct, plus_gros_topic, plus_gros_topic_pct.
    """
    rows: list[dict[str, Any]] = []
    for macro in MACRO_NAMES:
        stats = dict(macro_topic_counts.get(macro) or {})
        n_units = int(stats.get("n_units", 0))
        n_topics = int(stats.get("n_topics", 0))
        noise_rate = float(stats.get("noise_rate", 0.0))
        largest_share = float(stats.get("largest_topic_share", 0.0))
        tid = int(stats.get("largest_topic_id", -1))
        if tid < 0 and not assignments.empty:
            tid, _ = _dominant_topic_from_assignments(assignments, macro)
        label = _theme_label_for(themes, macro, tid) if tid >= 0 else ""
        if not label and tid >= 0:
            label = f"T{tid}"
        rows.append(
            {
                "macro": macro,
                "n_units": n_units,
                "n_topics": n_topics,
                "bruit_pct": round(100.0 * noise_rate, 1),
                "plus_gros_topic": label,
                "plus_gros_topic_pct": round(100.0 * largest_share, 1),
            }
        )
    return pd.DataFrame(rows)


def format_macro_topic_stats_display(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes pour affichage notebook (FR)."""
    if df.empty:
        return df
    rename = {
        "macro": "Macro",
        "n_units": "Unités",
        "n_topics": "Topics",
        "bruit_pct": "Bruit",
        "plus_gros_topic": "Plus gros topic",
        "plus_gros_topic_pct": "Part plus gros topic (%)",
    }
    out = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    for col in ("Bruit", "Part plus gros topic (%)"):
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
    return out
