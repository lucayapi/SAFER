"""Construction BERTopic (UMAP, HDBSCAN, c-TF-IDF) pour macro_transfer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from macro_transfer.representation import (
    build_representation_model,
    representation_enabled,
)

_PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_STOP_WORDS_FILE = _PACKAGE_DIR / "stop_metier.txt"
FRENCH_TOKEN_PATTERN = r"(?u)\b[a-zàâäæçéèêëîïôœùûüÿñ0-9]{2,}\b"


def load_stop_metier(path: Optional[Path] = None) -> List[str]:
    """Charge les stop words métier (une entrée par ligne, # commentaires)."""
    stop_path = Path(path) if path else DEFAULT_STOP_WORDS_FILE
    if not stop_path.is_file():
        raise FileNotFoundError(f"stop_metier introuvable : {stop_path}")
    words: List[str] = []
    with open(stop_path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            words.append(stripped.lower())
    if not words:
        raise ValueError(f"Aucun stop word dans {stop_path}")
    return words


def resolve_stop_words_file(cfg: Dict[str, Any], *, anchor: Optional[Path] = None) -> Path:
    """Résout stop_words_file depuis la config bertopic (null → défaut package)."""
    raw = cfg.get("stop_words_file")
    if raw is None or (isinstance(raw, str) and not str(raw).strip()):
        return DEFAULT_STOP_WORDS_FILE
    p = Path(str(raw))
    if p.is_file():
        return p
    if anchor is not None:
        candidate = anchor / p
        if candidate.is_file():
            return candidate
    candidate = _PACKAGE_DIR.parent / p
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"stop_words_file introuvable : {raw}")


def _parse_ngram_range(raw: Any) -> tuple[int, int]:
    if raw is None:
        return (1, 1)
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return (int(raw[0]), int(raw[1]))
    return (1, 1)


def umap_enabled(bertopic_cfg: Dict[str, Any]) -> bool:
    """
    True si une étape UMAP doit précéder HDBSCAN.

    Désactivé si ``umap: null``, ``umap: false``, ou ``umap.enabled: false``.
    Sinon (bloc de paramètres ou clé absente) : activé.
    """
    umap_raw = bertopic_cfg.get("umap", "__missing__")
    if umap_raw is None or umap_raw is False:
        return False
    if umap_raw == "__missing__":
        return True
    if isinstance(umap_raw, dict):
        return bool(umap_raw.get("enabled", True))
    return True


def _build_umap_model(umap_cfg: Dict[str, Any], *, random_state: int) -> Any:
    from umap import UMAP

    return UMAP(
        n_components=int(umap_cfg.get("n_components", 5)),
        n_neighbors=int(umap_cfg.get("n_neighbors", 15)),
        min_dist=float(umap_cfg.get("min_dist", 0.1)),
        metric=str(umap_cfg.get("metric", "cosine")),
        random_state=int(umap_cfg.get("random_state", random_state)),
    )


def build_bertopic_model(
    bertopic_cfg: Dict[str, Any],
    *,
    random_state: int = 42,
    anchor: Optional[Path] = None,
    representation_model: Any = None,
    macro: Optional[str] = None,
    corpus_id: Optional[str] = None,
):
    """Instancie BERTopic (UMAP optionnel), HDBSCAN, CountVectorizer et representation."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    min_topic_size = int(bertopic_cfg.get("min_topic_size", 10))
    nr_topics = bertopic_cfg.get("nr_topics")
    hdbscan_cfg = dict(bertopic_cfg.get("hdbscan") or {})
    vec_cfg = dict(bertopic_cfg.get("vectorizer") or {})

    stop_words = load_stop_metier(resolve_stop_words_file(bertopic_cfg, anchor=anchor))

    umap_model = None
    if umap_enabled(bertopic_cfg):
        raw_umap = bertopic_cfg.get("umap") or {}
        umap_cfg = {
            k: v for k, v in dict(raw_umap).items() if k != "enabled"
        }
        umap_model = _build_umap_model(umap_cfg, random_state=random_state)

    min_cluster_size = hdbscan_cfg.get("min_cluster_size")
    if min_cluster_size is None:
        min_cluster_size = min_topic_size
    hdbscan_kwargs: Dict[str, Any] = {
        "min_cluster_size": int(min_cluster_size),
        "metric": str(hdbscan_cfg.get("metric", "euclidean")),
        "cluster_selection_method": str(
            hdbscan_cfg.get("cluster_selection_method", "eom")
        ),
        "prediction_data": True,
    }
    min_samples = hdbscan_cfg.get("min_samples")
    if min_samples is not None:
        hdbscan_kwargs["min_samples"] = int(min_samples)
    hdbscan_model = HDBSCAN(**hdbscan_kwargs)

    ngram_range = _parse_ngram_range(
        vec_cfg.get("ngram_range", bertopic_cfg.get("n_gram_range"))
    )
    vectorizer_kwargs: Dict[str, Any] = {
        "stop_words": stop_words,
        "token_pattern": FRENCH_TOKEN_PATTERN,
        "min_df": int(vec_cfg.get("min_df", 1)),
        "ngram_range": ngram_range,
    }
    max_df = vec_cfg.get("max_df")
    if max_df is not None:
        vectorizer_kwargs["max_df"] = float(max_df)
    vectorizer_model = CountVectorizer(**vectorizer_kwargs)

    rep_model = representation_model
    if rep_model is None and representation_enabled(bertopic_cfg):
        rep_cfg = dict(bertopic_cfg.get("representation") or {})
        rep_cfg.setdefault("enabled", True)
        rep_model = build_representation_model(
            rep_cfg, macro=macro, corpus_id=corpus_id, anchor=anchor
        )

    n_gram_range = _parse_ngram_range(bertopic_cfg.get("n_gram_range"))

    return BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=rep_model,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        top_n_words=int(bertopic_cfg.get("top_n_words", 10)),
        n_gram_range=n_gram_range,
        calculate_probabilities=bool(bertopic_cfg.get("calculate_probabilities", True)),
        verbose=bool(bertopic_cfg.get("verbose", False)),
    )


def fit_bertopic_subset(
    texts: Sequence[str],
    embeddings: np.ndarray,
    bertopic_cfg: Dict[str, Any],
    *,
    random_state: int = 42,
    anchor: Optional[Path] = None,
    macro: Optional[str] = None,
    corpus_id: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Fit BERTopic sur un sous-corpus (embeddings pré-calculés).

    Retourne (topic_ids, confidences, model).
    """
    model = build_bertopic_model(
        bertopic_cfg,
        random_state=random_state,
        anchor=anchor,
        macro=macro,
        corpus_id=corpus_id,
    )
    topics, probs = model.fit_transform(list(texts), embeddings)
    topic_arr = np.asarray(topics, dtype=np.int64)
    if probs is not None and hasattr(probs, "shape") and len(probs.shape) == 2:
        conf = np.asarray(probs).max(axis=1)
    else:
        conf = np.ones(len(texts), dtype=np.float64)
    return topic_arr, conf, model


def format_topic_words(
    model: Any,
    topic_id: int,
    *,
    top_k: int = 12,
) -> str:
    """Mots du topic via c-TF-IDF BERTopic (``get_topic``)."""
    if topic_id < 0:
        return ""
    words = model.get_topic(int(topic_id))
    if not words:
        return ""
    parts = []
    for word, _score in words[:top_k]:
        if word:
            parts.append(str(word))
    return " ".join(parts)


def topic_label_from_model(model: Any, topic_id: int) -> str:
    """Libellé LLM (colonne Name / topic_labels_) après representation OpenAI."""
    if topic_id < 0:
        return ""
    labels = getattr(model, "topic_labels_", None)
    if isinstance(labels, dict) and topic_id in labels:
        name = labels[topic_id]
        if name and str(name).strip():
            return str(name).strip()
    try:
        info = model.get_topic_info()
        if info is not None and len(info) > 0 and "Topic" in info.columns:
            row = info.loc[info["Topic"] == topic_id]
            if len(row) and "Name" in row.columns:
                val = row.iloc[0]["Name"]
                if val is not None and str(val).strip():
                    return str(val).strip()
    except Exception:
        pass
    return ""
