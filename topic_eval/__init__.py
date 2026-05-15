"""Évaluation et comparaison de qualité thématique (MALT, BERTopic, KMeans + c-TF-IDF)."""

from .compare_topics import (
    build_malt_topics_df,
    build_topics_by_macro_qualitative,
    check_malt_export_files,
    load_embeddings_and_metadata,
    write_topic_comparison_report_md,
)
from .paths import find_repo_root, resolve_repo_path
from .topic_cleaning import (
    clean_top_words,
    load_domain_stopwords,
    normalize_token,
    preprocess_for_ctfidf,
)

__all__ = [
    "load_domain_stopwords",
    "normalize_token",
    "clean_top_words",
    "preprocess_for_ctfidf",
    "find_repo_root",
    "resolve_repo_path",
    "check_malt_export_files",
    "load_embeddings_and_metadata",
    "build_malt_topics_df",
    "build_topics_by_macro_qualitative",
    "write_topic_comparison_report_md",
]
