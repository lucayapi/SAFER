"""c-TF-IDF par classe (topic) pour extraire les mots représentatifs."""

from __future__ import annotations

from typing import Callable, List, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


def _c_tf_idf_from_counts(count_matrix: sparse.csr_matrix) -> np.ndarray:
    """count_matrix: (n_topics, n_vocab) — scores c-TF-IDF (dense).

    Implémentation dense uniquement : compatible toutes versions SciPy
    (évite ``sparse + scalaire`` et ``sparse array`` non supportés).
    """
    count_matrix = count_matrix.tocsr().astype(np.float64)
    n_topics, _ = count_matrix.shape
    if n_topics == 0:
        return np.zeros((0, 0))
    X = count_matrix.toarray()
    row_sum = X.sum(axis=1, keepdims=True) + 1e-12
    tf = X / row_sum
    df = (X > 0).sum(axis=0).astype(np.float64) + 1e-12
    idf = np.log(1.0 + float(n_topics) / df)
    return tf * idf


def top_words_ctfidf(
    texts_per_topic: Sequence[Sequence[str]],
    n_words: int = 10,
    min_df: int = 1,
    max_df: float = 0.95,
    preprocessor: Callable[[str], str] | None = None,
) -> List[List[str]]:
    """
    texts_per_topic[i] = liste de documents (strings) du topic i.
    Retourne une liste de listes de tokens (vocab sklearn, non lemmatisé).
    """
    flat_labels: List[int] = []
    flat_docs: List[str] = []
    for topic_idx, docs in enumerate(texts_per_topic):
        for d in docs:
            flat_labels.append(topic_idx)
            if preprocessor:
                flat_docs.append(preprocessor(str(d)))
            else:
                flat_docs.append(str(d))

    if not flat_docs:
        return []

    vectorizer = CountVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[a-zàâäæçéèêëîïôœùûüÿñ]{2,}\b",
        min_df=min_df,
        max_df=max_df,
    )
    try:
        X = vectorizer.fit_transform(flat_docs)
    except ValueError:
        return [[] for _ in texts_per_topic]

    n_topics = len(texts_per_topic)
    n_vocab = X.shape[1]
    counts = sparse.lil_matrix((n_topics, n_vocab), dtype=np.float64)
    X_csr = X.tocsr()
    indptr = X_csr.indptr
    indices = X_csr.indices
    data = X_csr.data
    for row_idx, lab in enumerate(flat_labels):
        start, end = indptr[row_idx], indptr[row_idx + 1]
        cols = indices[start:end]
        vals = data[start:end]
        for c, v in zip(cols, vals):
            counts[lab, c] += float(v)

    counts_csr = counts.tocsr()
    scores = _c_tf_idf_from_counts(counts_csr)
    terms = np.asarray(vectorizer.get_feature_names_out())
    result: List[List[str]] = []
    for i in range(n_topics):
        if scores.shape[0] <= i:
            result.append([])
            continue
        row = np.asarray(scores[i, :]).ravel()
        if row.size == 0:
            result.append([])
            continue
        top_idx = np.argsort(-row)[: max(1, n_words)]
        result.append([str(terms[j]) for j in top_idx if row[j] > 0])
    return result
