"""BERTopic intra-macro avec embeddings pré-calculés."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from macro_transfer.constants import MACRO_NAMES
from macro_transfer.topics_export import export_themes_by_macro


def _fit_bertopic_subset(
    texts: list[str],
    embeddings: np.ndarray,
    *,
    min_topic_size: int = 10,
    nr_topics: Optional[int] = None,
    random_state: int = 42,
) -> tuple[np.ndarray, Any]:
    from bertopic import BERTopic

    model = BERTopic(
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        calculate_probabilities=True,
        verbose=False,
    )
    topics, probs = model.fit_transform(texts, embeddings)
    topic_arr = np.asarray(topics, dtype=np.int64)
    if probs is not None and hasattr(probs, "shape") and len(probs.shape) == 2:
        conf = np.asarray(probs).max(axis=1)
    else:
        conf = np.ones(len(texts), dtype=np.float64)
    return topic_arr, conf


def fit_bertopic_per_macro(
    z: np.ndarray,
    meta: pd.DataFrame,
    gating: pd.DataFrame,
    *,
    method: str = "bertopic",
    confidence_threshold: float = 0.5,
    min_topic_size: int = 10,
    nr_topics: Optional[int] = None,
    random_state: int = 42,
    output_dir: Path,
    sentence_col: str = "sentence",
    top_k_words: int = 12,
    top_k_sentences: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """BERTopic par macro sur sous-ensembles filtrés (non ambigus)."""
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
        texts = meta.iloc[idx][sentence_col].astype(str).tolist()
        emb_sub = z[idx]

        if len(texts) < max(3, min_topic_size):
            for doc_idx in idx:
                assign_rows.append(
                    {
                        "doc_idx": int(doc_idx),
                        "macro": macro,
                        "topic_id": 0,
                        "prob": 1.0,
                        "p_mk": float(p_macro[doc_idx, mi]),
                    }
                )
            continue

        try:
            topic_ids, conf = _fit_bertopic_subset(
                texts,
                emb_sub,
                min_topic_size=min_topic_size,
                nr_topics=nr_topics,
                random_state=random_state,
            )
        except Exception:
            topic_ids = np.zeros(len(texts), dtype=np.int64)
            conf = np.ones(len(texts), dtype=np.float64)

        for local_j, doc_idx in enumerate(idx):
            tid = int(topic_ids[local_j])
            if tid < 0:
                tid = -1
            prob = float(conf[local_j]) if local_j < len(conf) else 1.0
            pm = float(p_macro[doc_idx, mi] * prob) if tid >= 0 else 0.0
            assign_rows.append(
                {
                    "doc_idx": int(doc_idx),
                    "macro": macro,
                    "topic_id": tid,
                    "prob": prob,
                    "p_mk": pm,
                }
            )

        sub_assign = pd.DataFrame(
            [
                {
                    "doc_idx": int(doc_idx),
                    "m_hat": macro,
                    "topic_id": int(topic_ids[local_j]) if topic_ids[local_j] >= 0 else -1,
                }
                for local_j, doc_idx in enumerate(idx)
            ]
        )
        valid = sub_assign["topic_id"] >= 0
        if valid.any():
            themes = export_themes_by_macro(
                sub_assign.loc[valid],
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
