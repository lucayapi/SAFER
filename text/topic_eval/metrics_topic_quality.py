"""Métriques de qualité thématique : diversité, redondance, couverture, C_v, NPMI."""

from __future__ import annotations

from typing import Callable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .topic_cleaning import clean_top_words, tokenize_for_coherence


def topic_diversity_score(topics_token_lists: Sequence[Sequence[str]], n_top_words: int) -> float:
    lists = [list(t)[:n_top_words] for t in topics_token_lists]
    n_topics = len(lists)
    if n_topics == 0:
        return 0.0
    denom = float(n_topics * max(1, n_top_words))
    flat: List[str] = []
    for t in lists:
        flat.extend(t)
    if not flat:
        return 0.0
    return len(set(flat)) / denom


def redundancy_from_diversity(diversity: float) -> float:
    return 1.0 - float(diversity)


def mean_pairwise_jaccard_top_words(topics_token_lists: Sequence[Sequence[str]], n_top_words: int) -> float:
    sets = [set(t[:n_top_words]) for t in topics_token_lists if len(t) >= 1]
    if len(sets) < 2:
        return 0.0
    vals: List[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            union = len(a | b) or 1
            vals.append(len(a & b) / union)
    return float(np.mean(vals)) if vals else 0.0


def is_bertopic_outlier_row(topic_id) -> bool:
    s = str(topic_id)
    if s == "-1":
        return True
    if s.endswith("_-1"):
        return True
    return False


def coverage_from_assignment(
    topics_df: pd.DataFrame,
    total_docs: int,
    outlier_fn: Callable[[pd.Series], bool] | None = None,
) -> float:
    if total_docs <= 0:
        return 0.0
    assigned = 0
    for _, r in topics_df.iterrows():
        if outlier_fn is not None and outlier_fn(r):
            continue
        assigned += int(r.get("n_docs", 0))
    return assigned / float(total_docs)


def parse_topic_words_cell(top_words: str, stopwords_domain: Set[str], max_words: int) -> List[str]:
    if not isinstance(top_words, str) or not top_words.strip():
        return []
    raw = top_words.replace("||", " ").split()
    return clean_top_words(raw, stopwords_domain)[:max_words]


def _topics_intersect_dictionary(
    topics_words: Sequence[Sequence[str]],
    token2id: dict,
) -> List[List[str]]:
    """Ne garde que les tokens présents dans le dictionnaire Gensim (sinon CoherenceModel lève ValueError)."""
    out: List[List[str]] = []
    for t in topics_words:
        w = [str(x) for x in t if str(x) in token2id]
        if len(w) >= 2:
            out.append(w)
    return out


def compute_cv_coherence(
    topics_words: Sequence[Sequence[str]],
    tokenized_texts: Sequence[Sequence[str]],
) -> float:
    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel
    except ImportError as e:
        raise ImportError("Installez gensim pour C_v : pip install gensim") from e

    topics = [list(t) for t in topics_words if len(t) >= 2]
    if not topics:
        return float("nan")
    texts = [list(t) for t in tokenized_texts if t]
    if not texts:
        return float("nan")
    # Inclure les listes de mots des topics comme « pseudo-documents » pour que le
    # dictionnaire contienne les top_words (sinon c-TF-IDF MALT peut ne pas recouper
    # le vocabulaire issu seulement des phrases tokenisées → C_v / NPMI en NaN).
    pseudo_docs = [[w for w in t] for t in topics]
    dictionary = Dictionary(texts + pseudo_docs)
    dictionary.filter_extremes(no_below=1, no_above=0.95)
    if len(dictionary) == 0:
        return float("nan")
    topics = _topics_intersect_dictionary(topics, dictionary.token2id)
    if len(topics) < 1:
        return float("nan")
    corpus = [dictionary.doc2bow(t) for t in texts]
    try:
        cm = CoherenceModel(
            topics=topics,
            texts=texts,
            dictionary=dictionary,
            corpus=corpus,
            coherence="c_v",
            topn=min(10, max(len(t) for t in topics)),
        )
        return float(cm.get_coherence())
    except (ValueError, ZeroDivisionError, RuntimeError):
        return float("nan")


def compute_npmi_coherence(
    topics_words: Sequence[Sequence[str]],
    tokenized_texts: Sequence[Sequence[str]],
) -> float:
    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel
    except ImportError as e:
        raise ImportError("Installez gensim pour NPMI : pip install gensim") from e

    topics = [list(t) for t in topics_words if len(t) >= 2]
    if not topics:
        return float("nan")
    texts = [list(t) for t in tokenized_texts if t]
    if not texts:
        return float("nan")
    # Inclure les listes de mots des topics comme « pseudo-documents » pour que le
    # dictionnaire contienne les top_words (sinon c-TF-IDF MALT peut ne pas recouper
    # le vocabulaire issu seulement des phrases tokenisées → C_v / NPMI en NaN).
    pseudo_docs = [[w for w in t] for t in topics]
    dictionary = Dictionary(texts + pseudo_docs)
    dictionary.filter_extremes(no_below=1, no_above=0.95)
    if len(dictionary) == 0:
        return float("nan")
    topics = _topics_intersect_dictionary(topics, dictionary.token2id)
    if len(topics) < 1:
        return float("nan")
    corpus = [dictionary.doc2bow(t) for t in texts]
    try:
        cm = CoherenceModel(
            topics=topics,
            texts=texts,
            dictionary=dictionary,
            corpus=corpus,
            coherence="c_npmi",
            topn=min(10, max(len(t) for t in topics)),
        )
        return float(cm.get_coherence())
    except (ValueError, ZeroDivisionError, RuntimeError):
        return float("nan")


def _filter_topics_df(
    topics_df: pd.DataFrame,
    min_topic_size: int,
    apply_size_filter: bool,
) -> pd.DataFrame:
    out = topics_df.copy()
    if "topic_id" in out.columns:
        out = out.loc[~out["topic_id"].map(is_bertopic_outlier_row)].copy()
    if apply_size_filter and "n_docs" in out.columns:
        out = out.loc[out["n_docs"] >= min_topic_size].copy()
    return out


def compute_topic_quality_metrics(
    topics_df: pd.DataFrame,
    docs: Sequence[str],
    method_name: str,
    stopwords_domain: Set[str],
    n_top_words: int = 10,
    min_topic_size: int = 15,
    apply_min_topic_filter: bool = True,
) -> pd.DataFrame:
    df_metric = _filter_topics_df(topics_df, min_topic_size, apply_min_topic_filter)
    tokenized = [tokenize_for_coherence(str(d), stopwords_domain) for d in docs]

    topics_words_all: List[List[str]] = []
    topics_words_coh: List[List[str]] = []
    for _, row in df_metric.iterrows():
        words = parse_topic_words_cell(str(row.get("top_words", "")), stopwords_domain, n_top_words)
        topics_words_all.append(words[:n_top_words])
        if len(words) >= 2:
            topics_words_coh.append(words)

    n_all_rows = (
        len(topics_df.loc[~topics_df["topic_id"].map(is_bertopic_outlier_row)])
        if "topic_id" in topics_df.columns
        else len(topics_df)
    )
    n_valid_topics = len(df_metric)
    total_docs = len(docs)

    diversity = topic_diversity_score(topics_words_all, n_top_words)
    redundancy = redundancy_from_diversity(diversity)
    jacc = mean_pairwise_jaccard_top_words(topics_words_all, n_top_words)

    cov = coverage_from_assignment(
        topics_df,
        total_docs,
        lambda r: is_bertopic_outlier_row(r.get("topic_id", "")),
    )

    cv = float("nan")
    npmi = float("nan")
    if topics_words_coh:
        try:
            cv = compute_cv_coherence(topics_words_coh, tokenized)
        except ImportError:
            cv = float("nan")
        try:
            npmi = compute_npmi_coherence(topics_words_coh, tokenized)
        except ImportError:
            npmi = float("nan")

    sizes = [int(r.get("n_docs", 0)) for _, r in df_metric.iterrows()]
    mean_sz = float(np.mean(sizes)) if sizes else 0.0
    med_sz = float(np.median(sizes)) if sizes else 0.0

    return pd.DataFrame(
        [
            {
                "method": method_name,
                "n_topics": int(n_all_rows),
                "n_valid_topics": int(n_valid_topics),
                "coverage": cov,
                "cv": cv,
                "npmi": npmi,
                "topic_diversity": diversity,
                "redundancy": redundancy,
                "mean_pairwise_jaccard": jacc,
                "mean_topic_size": mean_sz,
                "median_topic_size": med_sz,
                "min_topic_filter_applied": apply_min_topic_filter,
            }
        ]
    )


def build_metrics_both_filters(
    topics_df: pd.DataFrame,
    docs: Sequence[str],
    method_name: str,
    stopwords_domain: Set[str],
    n_top_words: int,
    min_topic_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    main = compute_topic_quality_metrics(
        topics_df, docs, method_name, stopwords_domain, n_top_words, min_topic_size, True
    )
    raw = compute_topic_quality_metrics(
        topics_df, docs, method_name, stopwords_domain, n_top_words, min_topic_size, False
    )
    return main, raw


def macro_level_metrics(
    topics_df: pd.DataFrame,
    docs: Sequence[str],
    doc_macro_col: str,
    meta_aligned: pd.DataFrame,
    method_name: str,
    stopwords_domain: Set[str],
    n_top_words: int,
    min_topic_size: int,
) -> pd.DataFrame:
    """Métriques par macro (documents du corpus filtrés par macro du doc)."""
    rows = []
    macros = ["A0", "A1", "B", "C"]
    for macro in macros:
        mask = meta_aligned[doc_macro_col].astype(str) == macro
        idx = np.where(mask.values)[0]
        sub_docs = [docs[i] for i in idx]
        sub_topics = topics_df.loc[topics_df["macro"].astype(str) == macro].copy()
        if sub_topics.empty or not sub_docs:
            rows.append(
                {
                    "method": method_name,
                    "macro": macro,
                    "n_docs": int(mask.sum()),
                    "n_topics": 0,
                    "coverage_macro": 0.0,
                    "mean_topic_size": 0.0,
                    "cv_macro": float("nan"),
                    "npmi_macro": float("nan"),
                    "diversity_macro": 0.0,
                }
            )
            continue
        m = compute_topic_quality_metrics(
            sub_topics, sub_docs, method_name, stopwords_domain, n_top_words, min_topic_size, True
        )
        rows.append(
            {
                "method": method_name,
                "macro": macro,
                "n_docs": int(mask.sum()),
                "n_topics": int(m["n_topics"].iloc[0]),
                "coverage_macro": float(m["coverage"].iloc[0]),
                "mean_topic_size": float(m["mean_topic_size"].iloc[0]),
                "cv_macro": float(m["cv"].iloc[0]),
                "npmi_macro": float(m["npmi"].iloc[0]),
                "diversity_macro": float(m["topic_diversity"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)
