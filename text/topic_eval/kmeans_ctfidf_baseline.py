"""Baseline KMeans (ou MiniBatch) intra-macro + mots représentatifs par c-TF-IDF."""

from __future__ import annotations

from typing import List, Sequence, Set

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans

from .ctfidf import top_words_ctfidf
from .topic_cleaning import clean_top_words, preprocess_for_ctfidf


def _top_sentences_centroid(
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


def run_kmeans_ctfidf_intra_macro(
    docs: Sequence[str],
    embeddings: np.ndarray,
    macro_labels: Sequence[str],
    n_clusters_per_macro: int,
    stopwords_domain: Set[str],
    random_state: int = 42,
    n_top_words: int = 10,
    n_representative_sentences: int = 8,
    use_minibatch_if_n: int = 8000,
) -> pd.DataFrame:
    docs = list(docs)
    emb = np.asarray(embeddings, dtype=np.float64)
    macros = np.asarray(list(macro_labels))
    rows: List[dict] = []

    for macro in ["A0", "A1", "B", "C"]:
        m = macros == macro
        idx = np.where(m)[0]
        n_macro = len(idx)
        if n_macro < 3:
            continue
        k = min(int(n_clusters_per_macro), n_macro)
        if k < 2:
            k = 1
        sub_docs = [docs[i] for i in idx]
        sub_e = emb[idx]
        km_model: KMeans | MiniBatchKMeans
        if n_macro > use_minibatch_if_n:
            km_model = MiniBatchKMeans(
                n_clusters=k,
                random_state=random_state,
                batch_size=1024,
                n_init="auto",
            )
        else:
            km_model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km_model.fit_predict(sub_e)

        for c in range(k):
            mask = labels == c
            if not np.any(mask):
                continue
            sents = [sub_docs[j] for j in np.where(mask)[0]]
            sub_mat = sub_e[mask]
            centroid = km_model.cluster_centers_[c]
            tw_lists = top_words_ctfidf([sents], n_words=n_top_words, preprocessor=preprocess_for_ctfidf)
            raw_words = tw_lists[0] if tw_lists else []
            top_w = clean_top_words(raw_words, stopwords_domain)[:n_top_words]
            top_s = _top_sentences_centroid(sents, sub_mat, centroid, n_representative_sentences)
            dists = np.linalg.norm(sub_mat - centroid.reshape(1, -1), axis=1)
            mean_conf = float(1.0 / (1.0 + float(np.mean(dists)))) if dists.size else 0.0
            n_d = int(mask.sum())
            rows.append(
                {
                    "method": "KMeans intra-macro + c-TF-IDF",
                    "topic_id": f"{macro}_km_{c}",
                    "macro": macro,
                    "n_docs": n_d,
                    "coverage_docs": n_d / max(1, len(docs)),
                    "top_words": " ".join(top_w),
                    "top_sentences": " || ".join(top_s),
                    "mean_confidence": mean_conf,
                    "source": "KMeans intra-macro + c-TF-IDF",
                }
            )

    return pd.DataFrame(rows)
