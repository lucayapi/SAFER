"""Exports CSV détaillés BERTopic intra-macro."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from macro_transfer.bertopic_utils import (
    _ctfidf_word_scores,
    _is_llm_topic_representation,
    format_topic_words,
    topic_label_from_model,
)


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
            "largest_topic_id": -1,
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
    largest_idx = int(np.argmax(sizes))
    largest_topic_id = int(unique[largest_idx])
    largest = int(sizes.max())
    largest_share = float(largest / max(n_units, 1))
    probs = sizes / sizes.sum()
    diversity = float(-np.sum(probs * np.log(probs + 1e-12)))
    return {
        "n_units": n_units,
        "n_topics": n_topics,
        "n_noise": n_noise,
        "noise_rate": noise_rate,
        "largest_topic_id": largest_topic_id,
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
        words = _ctfidf_word_scores(model, int(tid), top_k=top_k) if model is not None else []
        if not words and model is not None:
            raw = model.get_topic(int(tid)) or []
            if not _is_llm_topic_representation(model, raw):
                words = [(str(w), float(s)) for w, s in raw[:top_k] if w]
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


def save_bertopic_document_datamap(
    model: Any,
    texts: Sequence[str],
    embeddings: np.ndarray,
    *,
    macro: str,
    output_path: Path,
    title: Optional[str] = None,
    custom_labels: bool = True,
    width: int = 1200,
    height: int = 1200,
) -> Optional[Path]:
    """
    Sauvegarde une carte DataMapPlot via ``BERTopic.visualize_document_datamap``.

    Utilise les embeddings du sous-ensemble macro (ex. Qwen brut) passés au fit.
    """
    if model is None or not texts:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib absent — DataMapPlot non exporté (macro=%s)", macro)
        return None
    try:
        fig = model.visualize_document_datamap(
            [str(t) for t in texts],
            embeddings=np.asarray(embeddings, dtype=np.float64),
            custom_labels=bool(custom_labels),
            title=title or f"BERTopic — macro {macro}",
            width=int(width),
            height=int(height),
        )
    except Exception as exc:
        logger.warning("DataMapPlot BERTopic échoué (macro=%s): %s", macro, exc)
        return None
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def bertopic_macro_model_path(macro_dir: Path) -> Path:
    return Path(macro_dir) / "bertopic_model.pkl"


def _bertopic_pickle_snapshot(model: Any) -> Dict[str, Any]:
    """Retire temporairement les attributs souvent non picklables."""
    snap: Dict[str, Any] = {}
    for attr in ("embedding_model", "representation_model"):
        if hasattr(model, attr):
            snap[attr] = getattr(model, attr)
            setattr(model, attr, None)
    vec = getattr(model, "vectorizer_model", None)
    if vec is not None and hasattr(vec, "stop_words_"):
        snap["vectorizer_stop_words_"] = vec.stop_words_
        vec.stop_words_ = None
    return snap


def _bertopic_pickle_restore(model: Any, snap: Dict[str, Any]) -> None:
    for attr in ("embedding_model", "representation_model"):
        if attr in snap:
            setattr(model, attr, snap[attr])
    if "vectorizer_stop_words_" in snap:
        vec = getattr(model, "vectorizer_model", None)
        if vec is not None:
            vec.stop_words_ = snap["vectorizer_stop_words_"]


def _joblib_save_bertopic_model(model: Any, path: Path) -> None:
    import joblib

    snap = _bertopic_pickle_snapshot(model)
    try:
        with open(path, "wb") as handle:
            joblib.dump(model, handle)
    finally:
        _bertopic_pickle_restore(model, snap)


def save_bertopic_macro_model(model: Any, macro_dir: Path) -> Optional[Path]:
    """Persiste le modèle BERTopic fité (pickle, sans embedding / representation)."""
    if model is None:
        return None
    macro_dir = Path(macro_dir)
    macro_dir.mkdir(parents=True, exist_ok=True)
    path = bertopic_macro_model_path(macro_dir)
    save_fn = getattr(model, "save", None)
    if callable(save_fn):
        for kwargs in (
            {"serialization": "pickle", "save_embedding_model": False},
            {"save_embedding_model": False},
            {},
        ):
            try:
                save_fn(str(path), **kwargs)
                if path.is_file():
                    return path
            except TypeError:
                continue
            except Exception as exc:
                logger.debug("BERTopic.save échoué (%s) : %s", kwargs, exc)
                break
    try:
        _joblib_save_bertopic_model(model, path)
    except Exception as exc:
        logger.warning("Sauvegarde modèle BERTopic échouée (%s): %s", macro_dir, exc)
        return None
    return path if path.is_file() else None


def load_bertopic_macro_model(macro_dir: Path) -> Any:
    path = bertopic_macro_model_path(macro_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Modèle BERTopic introuvable : {path}")
    try:
        from bertopic import BERTopic

        return BERTopic.load(str(path), embedding_model=None)
    except ImportError:
        pass
    except TypeError:
        try:
            from bertopic import BERTopic

            return BERTopic.load(str(path))
        except Exception:
            pass
    except Exception:
        pass
    import joblib

    with open(path, "rb") as handle:
        return joblib.load(handle)


def export_bertopic_datamaps_from_run(
    out_dir: str | Path,
    meta: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    macros: Optional[Sequence[str]] = None,
    text_col: str = "sentence",
    fig_dir: Optional[Path] = None,
    show_progress: bool = True,
    assignments_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Exporte les DataMapPlot par macro à partir des modèles sauvegardés (``bertopic/<macro>/``).

    Indépendant du fit BERTopic : nécessite ``diagnostics.save_model: true`` lors du fit.
    """
    from macro_transfer.bertopic_utils import bertopic_progress
    from macro_transfer.constants import MACRO_NAMES

    out_dir = Path(out_dir)
    macro_list = list(macros) if macros is not None else list(MACRO_NAMES)
    assign_path = Path(assignments_path) if assignments_path else out_dir / "topics_bertopic" / "assignments.csv"
    if not assign_path.is_file():
        raise FileNotFoundError(f"assignments.csv introuvable : {assign_path}")

    assignments = pd.read_csv(assign_path)
    meta_idx = meta.reset_index(drop=True)
    emb = np.asarray(embeddings, dtype=np.float64)
    fig_root = Path(fig_dir) if fig_dir is not None else out_dir / "figures"
    fig_root.mkdir(parents=True, exist_ok=True)
    per_macro_root = out_dir / "bertopic"

    saved: Dict[str, Path] = {}
    macro_iter: Any = macro_list
    if show_progress:
        from tqdm.auto import tqdm

        macro_iter = tqdm(macro_list, desc="DataMapPlot BERTopic", unit="macro")

    for macro in macro_iter:
        macro_s = str(macro)
        macro_dir = per_macro_root / macro_s
        model_path = bertopic_macro_model_path(macro_dir)
        if not model_path.is_file():
            if show_progress:
                bertopic_progress(f"  → {macro_s} : modèle absent ({model_path.name}), ignoré")
            continue

        sub = assignments.loc[assignments["macro"].astype(str) == macro_s]
        if sub.empty:
            continue
        doc_idx = pd.to_numeric(sub["doc_idx"], errors="coerce").dropna().astype(int).to_numpy()
        if len(doc_idx) == 0:
            continue
        if doc_idx.max() >= len(meta_idx) or doc_idx.min() < 0:
            raise ValueError(
                f"doc_idx hors limites pour macro {macro_s} (max={doc_idx.max()}, len(meta)={len(meta_idx)})"
            )

        texts = meta_idx.iloc[doc_idx][text_col].astype(str).tolist()
        emb_sub = emb[doc_idx]
        if show_progress:
            bertopic_progress(f"  → {macro_s} : DataMapPlot ({len(texts)} points)…")

        model = load_bertopic_macro_model(macro_dir)
        datamap_path = macro_dir / "datamap_topics.png"
        out_png = save_bertopic_document_datamap(
            model,
            texts,
            emb_sub,
            macro=macro_s,
            output_path=datamap_path,
        )
        if out_png is not None:
            fig_copy = fig_root / f"bertopic_datamap_{macro_s}.png"
            import shutil

            shutil.copy(out_png, fig_copy)
            saved[macro_s] = fig_copy
            if show_progress:
                bertopic_progress(f"  → {macro_s} : écrit {fig_copy}")

    return saved
