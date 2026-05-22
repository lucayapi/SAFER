"""BERTopic intra-macro avec embeddings pré-calculés (UMAP optionnel, HDBSCAN, c-TF-IDF)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from macro_transfer.bertopic_exports import (
    compute_topic_stats,
    write_macro_bertopic_exports,
)
from macro_transfer.bertopic_utils import (
    fit_bertopic_subset,
    format_topic_words,
    topic_label_from_model,
)
from macro_transfer.constants import MACRO_NAMES
from macro_transfer.topic_embeddings import build_topic_embeddings
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


def _append_bertopic_warning(
    warnings_path: Optional[Path],
    message: str,
) -> None:
    if warnings_path is None:
        return
    warnings_path = Path(warnings_path)
    warnings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(warnings_path, "a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def _resolve_topic_matrix(
    idx: np.ndarray,
    *,
    z: Optional[np.ndarray],
    embeddings_initial: Optional[np.ndarray],
    embeddings_adapted: Optional[np.ndarray],
    topic_embedding_cfg: Optional[Dict[str, Any]],
) -> np.ndarray:
    if embeddings_initial is not None:
        hi = np.asarray(embeddings_initial, dtype=np.float64)[idx]
        ha = None
        if embeddings_adapted is not None:
            ha = np.asarray(embeddings_adapted, dtype=np.float64)[idx]
        cfg = topic_embedding_cfg or {"mode": "initial", "alpha": 0.0, "normalize": True}
        return build_topic_embeddings(
            hi,
            ha,
            mode=str(cfg.get("mode", "initial")),
            alpha=float(cfg.get("alpha", 0.0)),
            normalize=bool(cfg.get("normalize", True)),
        )
    if z is None:
        raise ValueError("z ou embeddings_initial requis pour BERTopic intra-macro")
    return np.asarray(z, dtype=np.float64)[idx]


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
    output_dir: Path = None,
    sentence_col: str = "sentence",
    top_k_words: int = 12,
    top_k_sentences: int = 5,
    repo_anchor: Optional[Path] = None,
    corpus_id: Optional[str] = None,
    embeddings_initial: Optional[np.ndarray] = None,
    embeddings_adapted: Optional[np.ndarray] = None,
    topic_embedding_cfg: Optional[Dict[str, Any]] = None,
    per_macro_output_root: Optional[Path] = None,
    legacy_output_dir: Optional[Path] = None,
    run_output_root: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    BERTopic par macro sur sous-ensembles filtrés (non ambigus).

    Retourne (themes_all, assignments, bertopic_summary_partial).
    """
    if meta is None or gating is None:
        raise ValueError("meta et gating sont requis")
    legacy_dir = Path(legacy_output_dir or output_dir)
    legacy_dir.mkdir(parents=True, exist_ok=True)
    per_macro_root = Path(per_macro_output_root) if per_macro_output_root else legacy_dir.parent / "bertopic"
    per_macro_root.mkdir(parents=True, exist_ok=True)

    warnings_path = None
    if run_output_root is not None:
        warnings_path = Path(run_output_root) / "bertopic_warnings.txt"
        if warnings_path.is_file():
            warnings_path.unlink()

    cfg = dict(bertopic_cfg or {})
    cfg.setdefault("min_topic_size", min_topic_size)
    if nr_topics is not None:
        cfg["nr_topics"] = nr_topics
    cfg.setdefault("random_state", random_state)
    include_ambiguous = bool(cfg.get("include_ambiguous", False))
    diagnostics_cfg = dict(cfg.get("diagnostics") or {})
    n_rep_docs = int(diagnostics_cfg.get("n_representative_docs", 5))

    meta = meta.reset_index(drop=True)
    gating = gating.reset_index(drop=True)
    if len(meta) != len(gating):
        raise ValueError(
            f"meta ({len(meta)}) et gating ({len(gating)}) ont des longueurs différentes."
        )
    if z is not None and embeddings_initial is None:
        embeddings_initial = np.asarray(z, dtype=np.float64)

    prob_cols = [f"p_{m}" for m in MACRO_NAMES]
    p_macro = gating[prob_cols].to_numpy(dtype=np.float64)

    assign_rows: list[dict] = []
    theme_parts: list[pd.DataFrame] = []
    rs = int(cfg.get("random_state", random_state))
    macro_topic_counts: Dict[str, Any] = {}
    warnings_list: list[str] = []

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
        emb_sub = _resolve_topic_matrix(
            idx,
            z=z,
            embeddings_initial=embeddings_initial,
            embeddings_adapted=embeddings_adapted,
            topic_embedding_cfg=topic_embedding_cfg,
        )

        min_sz = int(
            (cfg.get("macro_params") or {}).get(macro, {}).get("min_topic_size", cfg.get("min_topic_size", min_topic_size))
        )
        if len(texts) < max(3, min_sz):
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
            macro_topic_counts[macro] = {
                "n_units": len(texts),
                "n_topics": 1,
                "noise_rate": 0.0,
                "largest_topic_share": 1.0,
            }
            continue

        try:
            topic_ids, conf, model = fit_bertopic_subset(
                texts,
                emb_sub,
                cfg,
                random_state=rs,
                anchor=repo_anchor,
                macro=macro,
                corpus_id=corpus_id,
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

        stats = compute_topic_stats(topic_ids, n_units=len(texts))
        macro_topic_counts[macro] = {
            "n_units": stats["n_units"],
            "n_topics": stats["n_topics"],
            "noise_rate": stats["noise_rate"],
            "largest_topic_share": stats["largest_topic_share"],
        }

        if macro in ("A0", "A1") and stats["n_topics"] <= 2:
            msg = (
                f"{macro} has only {stats['n_topics']} topics for {stats['n_units']} units. "
                "This suggests excessive compression or overly conservative HDBSCAN parameters. "
                "Consider using initial/mixed embeddings, lower min_cluster_size, lower min_samples, "
                "or cluster_selection_method='leaf'."
            )
            logger.warning(msg)
            warnings_list.append(msg)
            _append_bertopic_warning(warnings_path, msg)

        macro_dir = per_macro_root / macro
        emb_for_export = emb_sub
        if bool(diagnostics_cfg.get("save_representative_docs", True)):
            write_macro_bertopic_exports(
                macro_dir,
                model=model,
                macro=macro,
                topic_ids=topic_ids,
                confidences=conf,
                texts=texts,
                embeddings=emb_for_export,
                doc_indices=idx,
                meta=meta,
                config_extra={
                    "topic_embedding_mode": (topic_embedding_cfg or {}).get("mode"),
                    "topic_embedding_alpha": (topic_embedding_cfg or {}).get("alpha"),
                },
                top_k_words=top_k_words,
                n_representative_docs=n_rep_docs,
                bertopic_cfg=cfg,
            )

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
                emb_sub,
                method=method,
                sentence_col=sentence_col,
                top_k_words=top_k_words,
                top_k_sentences=top_k_sentences,
            )
            if theme_rows:
                themes = pd.DataFrame(theme_rows)
                themes.to_csv(legacy_dir / f"themes_macro_{macro}.csv", index=False)
                theme_parts.append(themes)

    assignments = pd.DataFrame(assign_rows)
    assignments.to_csv(legacy_dir / "assignments.csv", index=False)
    themes_all = pd.concat(theme_parts, ignore_index=True) if theme_parts else pd.DataFrame()
    if len(themes_all):
        themes_all.to_csv(legacy_dir / "themes_by_macro.csv", index=False)

    summary_partial = {
        "macro_topic_counts": macro_topic_counts,
        "warnings": warnings_list,
    }
    return themes_all, assignments, summary_partial
