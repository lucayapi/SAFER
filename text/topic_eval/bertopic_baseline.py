"""Baseline BERTopic appliqué séparément dans chaque macro (pseudo-labels p0)."""

from __future__ import annotations

from typing import List, Sequence, Set

import numpy as np
import pandas as pd

from .ctfidf import top_words_ctfidf
from .topic_cleaning import preprocess_for_ctfidf


def _top_sentences_near_embedding(
    sentences: Sequence[str],
    embeddings: np.ndarray,
    centroid: np.ndarray,
    top_k: int,
) -> List[str]:
    if len(sentences) == 0:
        return []
    c = centroid.reshape(1, -1)
    dists = np.linalg.norm(embeddings - c, axis=1)
    order = np.argsort(dists)[:top_k]
    return [str(sentences[i]) for i in order]


def _build_bertopic(BERTopic: type, cluster, min_topic_size: int, random_state: int):
    """Instancie ``BERTopic`` selon la version installée (arguments du constructeur variables)."""
    attempts: list[dict] = [
        {
            "hdbscan_model": cluster,
            "min_topic_size": min_topic_size,
            "calculate_probabilities": True,
            "verbose": False,
            "random_state": random_state,
        },
        {
            "hdbscan_model": cluster,
            "min_topic_size": min_topic_size,
            "calculate_probabilities": True,
            "verbose": False,
        },
        {
            "hdbscan_model": cluster,
            "min_topic_size": min_topic_size,
            "calculate_probabilities": True,
        },
        {"hdbscan_model": cluster, "min_topic_size": min_topic_size},
        {"hdbscan_model": cluster, "calculate_probabilities": True},
        {"hdbscan_model": cluster},
    ]
    last_err: TypeError | None = None
    for kw in attempts:
        try:
            return BERTopic(**kw)
        except TypeError as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    return BERTopic(hdbscan_model=cluster)


def run_bertopic_intra_macro(
    docs: Sequence[str],
    embeddings: np.ndarray,
    macro_labels: Sequence[str],
    stopwords_domain: Set[str],
    min_topic_size: int = 15,
    random_state: int = 42,
    n_top_words: int = 10,
    n_representative_sentences: int = 8,
) -> pd.DataFrame:
    """
    BERTopic indépendant par macro (A0, A1, B, C) sur les embeddings pré-calculés.
    """
    try:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
    except ImportError as e:
        raise ImportError(
            "BERTopic n'est pas installé. Installez-le avec : pip install bertopic"
        ) from e

    from .topic_cleaning import clean_top_words

    docs = list(docs)
    emb = np.asarray(embeddings, dtype=np.float64)
    macros = np.asarray(list(macro_labels))
    rows: List[dict] = []

    for macro in ["A0", "A1", "B", "C"]:
        m = macros == macro
        idx = np.where(m)[0]
        if len(idx) < max(min_topic_size, 3):
            continue
        sub_docs = [docs[i] for i in idx]
        sub_emb = emb[idx]
        cluster = HDBSCAN(
            min_cluster_size=max(2, min_topic_size // 2),
            min_samples=1,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        topic_model = _build_bertopic(BERTopic, cluster, min_topic_size, random_state)
        try:
            topics, probs = topic_model.fit_transform(sub_docs, embeddings=sub_emb)
        except TypeError:
            topics, probs = topic_model.fit_transform(sub_docs, sub_emb)
        except Exception:
            continue

        topic_info = topic_model.get_topic_info()
        centroids: dict[int, np.ndarray] = {}
        for _, trow in topic_info.iterrows():
            tid = int(trow["Topic"])
            if tid < 0:
                continue
            mask = np.asarray(topics) == tid
            if mask.any():
                centroids[tid] = sub_emb[mask].mean(axis=0)

        for _, trow in topic_info.iterrows():
            local_id = int(trow["Topic"])
            global_tid = f"{macro}_{local_id}"
            mask = np.asarray(topics) == local_id
            n_d = int(mask.sum())
            sents = [sub_docs[j] for j in np.where(mask)[0]]
            sub_e = sub_emb[mask]
            mean_conf = 1.0
            if probs is not None and hasattr(probs, "shape") and len(probs.shape) == 2 and local_id >= 0:
                pos = np.where(mask)[0]
                if pos.size:
                    mean_conf = float(np.mean(np.max(probs[pos], axis=1)))
            top_w_raw = []
            if local_id >= 0:
                try:
                    tw = topic_model.get_topic(local_id)
                    if tw:
                        top_w_raw = [w for w, _ in tw[: max(n_top_words * 3, 20)]]
                except Exception:
                    top_w_raw = []
            top_w = clean_top_words(top_w_raw, stopwords_domain)[:n_top_words]
            if not top_w and sents:
                tw_lists = top_words_ctfidf([sents], n_words=n_top_words, preprocessor=preprocess_for_ctfidf)
                if tw_lists:
                    top_w = clean_top_words(tw_lists[0], stopwords_domain)[:n_top_words]

            if local_id >= 0 and local_id in centroids:
                top_s = _top_sentences_near_embedding(sents, sub_e, centroids[local_id], n_representative_sentences)
            else:
                top_s = sents[:n_representative_sentences]

            rows.append(
                {
                    "method": "BERTopic intra-macro via p0",
                    "topic_id": global_tid,
                    "macro": macro,
                    "n_docs": n_d,
                    "coverage_docs": n_d / max(1, len(docs)),
                    "top_words": " ".join(top_w),
                    "top_sentences": " || ".join(top_s),
                    "mean_confidence": mean_conf if local_id >= 0 else 0.0,
                    "source": "BERTopic intra-macro via p0",
                }
            )

    return pd.DataFrame(rows)
