"""BERTopic intra-macro avec embeddings pré-calculés (UMAP, HDBSCAN, c-TF-IDF)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from macro_transfer.bertopic_utils import (
    fit_bertopic_subset,
    format_topic_words,
    topic_label_from_model,
)
from macro_transfer.constants import MACRO_NAMES
from scgm_text.topic_export import _top_sentences_by_distance


def _build_theme_rows(
    model: Any,
    macro: str,
    sub_assign: pd.DataFrame,
    meta: pd.DataFrame,
    z: np.ndarray,
    *,
    method: str,
    sentence_col: str,
    top_k_words: int,
    top_k_sentences: int,
) -> list[dict]:
    rows: list[dict] = []
    for topic_id in sorted(sub_assign["topic_id"].dropna().unique()):
        tid = int(topic_id)
        if tid < 0:
            continue
        sub_t = sub_assign.loc[sub_assign["topic_id"] == tid]
        idx = sub_t["doc_idx"].to_numpy(dtype=np.int64)
        subset = meta.iloc[idx]
        z_sub = z[idx]
        centroid = z_sub.mean(axis=0) if len(z_sub) else None
        sentences = subset[sentence_col].astype(str).tolist()
        top_words = format_topic_words(model, tid, top_k=top_k_words)
        theme_label = topic_label_from_model(model, tid)
        if centroid is not None and len(sentences) > 0:
            top_sentences = _top_sentences_by_distance(
                sentences, z_sub, centroid, top_k=top_k_sentences
            )
        else:
            top_sentences = " || ".join(sentences[:top_k_sentences])
        rows.append(
            {
                "method": method,
                "macro": macro,
                "topic_id": tid,
                "n_units": int(len(subset)),
                "top_words": top_words,
                "top_sentences": top_sentences,
                "theme_label": theme_label,
                "theme_title": theme_label,
            }
        )
    return rows


def fit_bertopic_per_macro(
    z: np.ndarray,
    meta: pd.DataFrame,
    gating: pd.DataFrame,
    *,
    method: str = "bertopic",
    bertopic_cfg: Optional[Dict[str, Any]] = None,
    min_topic_size: int = 10,
    nr_topics: Optional[int] = None,
    random_state: int = 42,
    output_dir: Path,
    sentence_col: str = "sentence",
    top_k_words: int = 12,
    top_k_sentences: int = 5,
    repo_anchor: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """BERTopic par macro sur sous-ensembles filtrés (non ambigus)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(bertopic_cfg or {})
    cfg.setdefault("min_topic_size", min_topic_size)
    if nr_topics is not None:
        cfg["nr_topics"] = nr_topics
    cfg.setdefault("random_state", random_state)
    include_ambiguous = bool(cfg.get("include_ambiguous", False))

    meta = meta.reset_index(drop=True)
    gating = gating.reset_index(drop=True)
    if len(meta) != len(gating):
        raise ValueError(
            f"meta ({len(meta)}) et gating ({len(gating)}) ont des longueurs différentes."
        )

    prob_cols = [f"p_{m}" for m in MACRO_NAMES]
    p_macro = gating[prob_cols].to_numpy(dtype=np.float64)

    assign_rows: list[dict] = []
    theme_parts: list[pd.DataFrame] = []
    rs = int(cfg.get("random_state", random_state))

    for macro in MACRO_NAMES:
        mi = MACRO_NAMES.index(macro)
        m_hat_mask = gating["m_hat"].astype(str) == macro
        if include_ambiguous:
            mask = m_hat_mask
        else:
            mask = m_hat_mask & (~gating["ambiguous"].astype(bool))
        if not mask.any():
            continue
        idx = np.where(mask.to_numpy())[0]
        texts = meta.iloc[idx][sentence_col].astype(str).tolist()
        emb_sub = z[idx]

        if len(texts) < max(3, int(cfg.get("min_topic_size", min_topic_size))):
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
            topic_ids, conf, model = fit_bertopic_subset(
                texts,
                emb_sub,
                cfg,
                random_state=rs,
                anchor=repo_anchor,
                macro=macro,
            )
        except Exception as exc:
            logger.exception(
                "BERTopic intra-macro échoué (macro=%s, n_units=%d)",
                macro,
                len(texts),
            )
            raise RuntimeError(
                f"BERTopic intra-macro échoué pour la macro {macro!r} "
                f"({len(texts)} unités non ambiguës). "
                "Voir la traceback ci-dessus (OpenAI, bertopic, mémoire, etc.)."
            ) from exc

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
        if valid.any() and model is not None:
            theme_rows = _build_theme_rows(
                model,
                macro,
                sub_assign.loc[valid],
                meta,
                z,
                method=method,
                sentence_col=sentence_col,
                top_k_words=top_k_words,
                top_k_sentences=top_k_sentences,
            )
            if theme_rows:
                themes = pd.DataFrame(theme_rows)
                themes.to_csv(output_dir / f"themes_macro_{macro}.csv", index=False)
                theme_parts.append(themes)

    assignments = pd.DataFrame(assign_rows)
    assignments.to_csv(output_dir / "assignments.csv", index=False)
    themes_all = pd.concat(theme_parts, ignore_index=True) if theme_parts else pd.DataFrame()
    if len(themes_all):
        themes_all.to_csv(output_dir / "themes_by_macro.csv", index=False)
    return themes_all, assignments
