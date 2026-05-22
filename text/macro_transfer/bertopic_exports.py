"""Exports CSV détaillés BERTopic intra-macro."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from macro_transfer.bertopic_utils import format_topic_words, topic_label_from_model


def compute_topic_stats(
    topic_ids: np.ndarray,
    *,
    n_units: int,
) -> Dict[str, Any]:
    """Statistiques topics (hors bruit -1)."""
    tid = np.asarray(topic_ids, dtype=np.int64)
    noise_mask = tid < 0
    n_noise = int(noise_mask.sum())
    noise_rate = float(n_noise / max(n_units, 1))
    valid = tid[~noise_mask]
    if len(valid) == 0:
        return {
            "n_units": n_units,
            "n_topics": 0,
            "n_noise": n_noise,
            "noise_rate": noise_rate,
            "largest_topic_size": 0,
            "largest_topic_share": 0.0,
            "median_topic_size": 0.0,
            "mean_topic_size": 0.0,
            "empty_topics": 0,
            "topic_diversity": 0.0,
        }
    unique, counts = np.unique(valid, return_counts=True)
    sizes = counts.astype(np.float64)
    n_topics = int(len(unique))
    largest = int(sizes.max())
    largest_share = float(largest / max(n_units, 1))
    probs = sizes / sizes.sum()
    diversity = float(-np.sum(probs * np.log(probs + 1e-12)))
    return {
        "n_units": n_units,
        "n_topics": n_topics,
        "n_noise": n_noise,
        "noise_rate": noise_rate,
        "largest_topic_size": largest,
        "largest_topic_share": largest_share,
        "median_topic_size": float(np.median(sizes)),
        "mean_topic_size": float(np.mean(sizes)),
        "empty_topics": 0,
        "topic_diversity": diversity,
    }


def topic_diversity_from_ids(topic_ids: np.ndarray) -> float:
    stats = compute_topic_stats(topic_ids, n_units=len(topic_ids))
    return float(stats["topic_diversity"])


def export_topic_keywords(
    model: Any,
    macro: str,
    topic_ids: Sequence[int],
    *,
    top_k: int = 12,
) -> pd.DataFrame:
    rows: list[dict] = []
    for tid in sorted(set(int(t) for t in topic_ids if int(t) >= 0)):
        words = model.get_topic(int(tid)) if model is not None else []
        for rank, item in enumerate(words[:top_k], start=1):
            if not item:
                continue
            word, score = item[0], item[1] if len(item) > 1 else 0.0
            rows.append(
                {
                    "macro": macro,
                    "topic_id": tid,
                    "rank": rank,
                    "word": str(word),
                    "score": float(score),
                }
            )
    return pd.DataFrame(rows)


def export_topics_overview(
    model: Any,
    macro: str,
    topic_ids: np.ndarray,
    texts: Sequence[str],
    *,
    n_units: int,
    top_k_words: int = 12,
) -> pd.DataFrame:
    stats_base = compute_topic_stats(topic_ids, n_units=n_units)
    rows: list[dict] = []
    tid_arr = np.asarray(topic_ids, dtype=np.int64)
    for tid in sorted(set(int(t) for t in tid_arr if int(t) >= 0)):
        mask = tid_arr == tid
        size = int(mask.sum())
        share = float(size / max(n_units, 1))
        top_words = format_topic_words(model, tid, top_k=top_k_words) if model else ""
        rep_short = ""
        if mask.any():
            rep_short = str(texts[int(np.where(mask)[0][0])])[:200]
        rows.append(
            {
                "topic_id": tid,
                "topic_size": size,
                "topic_share": share,
                "top_words": top_words,
                "representative_doc_short": rep_short,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["topic_id", "topic_size", "topic_share", "top_words", "representative_doc_short"]
        )
    return pd.DataFrame(rows)


def export_representative_docs(
    model: Any,
    macro: str,
    topic_ids: np.ndarray,
    texts: Sequence[str],
    embeddings: np.ndarray,
    *,
    n_docs: int = 5,
) -> pd.DataFrame:
    from scgm_text.topic_export import _top_sentences_by_distance

    rows: list[dict] = []
    tid_arr = np.asarray(topic_ids, dtype=np.int64)
    emb = np.asarray(embeddings, dtype=np.float64)
    for tid in sorted(set(int(t) for t in tid_arr if int(t) >= 0)):
        mask = tid_arr == tid
        idx_local = np.where(mask)[0]
        if len(idx_local) == 0:
            continue
        z_sub = emb[idx_local]
        centroid = z_sub.mean(axis=0)
        sentences = [str(texts[i]) for i in idx_local]
        ranked = _top_sentences_by_distance(sentences, z_sub, centroid, top_k=n_docs)
        for rank, sent in enumerate(ranked.split(" || ")[:n_docs], start=1):
            rows.append(
                {
                    "macro": macro,
                    "topic_id": tid,
                    "doc_rank": rank,
                    "doc_text": sent.strip(),
                }
            )
    return pd.DataFrame(rows)


def export_topic_assignments(
    macro: str,
    doc_indices: np.ndarray,
    topic_ids: np.ndarray,
    confidences: np.ndarray,
    texts: Sequence[str],
    meta: pd.DataFrame,
    *,
    unit_id_col: Optional[str] = None,
    accident_id_col: Optional[str] = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for local_j, doc_idx in enumerate(doc_indices):
        row: dict = {
            "unit_id": doc_idx,
            "macro": macro,
            "topic_id": int(topic_ids[local_j]),
            "topic_probability": float(confidences[local_j]) if local_j < len(confidences) else None,
            "text": str(texts[local_j]),
        }
        if unit_id_col and unit_id_col in meta.columns:
            row["unit_id"] = meta.iloc[int(doc_idx)][unit_id_col]
        if accident_id_col and accident_id_col in meta.columns:
            row["accident_id"] = meta.iloc[int(doc_idx)][accident_id_col]
        rows.append(row)
    return pd.DataFrame(rows)


def write_macro_bertopic_exports(
    output_dir: Path,
    *,
    model: Any,
    macro: str,
    topic_ids: np.ndarray,
    confidences: np.ndarray,
    texts: Sequence[str],
    embeddings: np.ndarray,
    doc_indices: np.ndarray,
    meta: pd.DataFrame,
    config_extra: Optional[Dict[str, Any]] = None,
    top_k_words: int = 12,
    n_representative_docs: int = 5,
    bertopic_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Écrit tous les CSV détaillés sous output_dir."""
    from macro_transfer.bertopic_utils import save_macro_bertopic_config

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_units = len(texts)
    stats = compute_topic_stats(topic_ids, n_units=n_units)

    overview = export_topics_overview(
        model, macro, topic_ids, texts, n_units=n_units, top_k_words=top_k_words
    )
    overview.to_csv(output_dir / "topics_overview.csv", index=False)

    kw = export_topic_keywords(model, macro, topic_ids, top_k=top_k_words)
    kw.to_csv(output_dir / "topic_keywords.csv", index=False)

    rep = export_representative_docs(
        model,
        macro,
        topic_ids,
        texts,
        embeddings,
        n_docs=n_representative_docs,
    )
    rep.to_csv(output_dir / "representative_docs.csv", index=False)

    assign = export_topic_assignments(
        macro,
        doc_indices,
        topic_ids,
        confidences,
        texts,
        meta,
        accident_id_col="accident_id" if "accident_id" in meta.columns else None,
    )
    assign.to_csv(output_dir / "topic_assignments.csv", index=False)

    if bertopic_cfg is not None:
        save_macro_bertopic_config(
            macro,
            bertopic_cfg,
            output_dir / "config_used.json",
            extra=config_extra,
        )

    return stats
