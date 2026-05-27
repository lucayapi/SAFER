"""Helpers BERTopic pour le pipeline TPN full encoder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from macro_transfer.bertopic_grid import run_macro_bertopic_grid_search
from macro_transfer.intra_bertopic import fit_bertopic_per_macro
from macro_transfer.macro_compression import compute_macro_compression_diagnostics
from macro_transfer.topic_embeddings import resolve_topic_embedding_cfg
from macro_transfer.topics_export import build_macro_topic_test_table, summarize_topics_by_macro
from safer_core.paths import resolve_repo_path


def merge_bertopic_cfg(
    bertopic_cfg: Dict[str, Any],
    macro_topic_config_path: Optional[str],
    repo_anchor: Path,
) -> Dict[str, Any]:
    cfg = dict(bertopic_cfg or {})
    if not macro_topic_config_path:
        return cfg
    extra_path = resolve_repo_path(macro_topic_config_path, repo_root=repo_anchor)
    with open(extra_path, encoding="utf-8") as f:
        extra = yaml.safe_load(f) or {}
    if "macro_params" in extra:
        cfg["macro_params"] = {**(cfg.get("macro_params") or {}), **extra["macro_params"]}
    for key in ("embedding_space", "grid_search", "diagnostics", "default_params"):
        if key in extra and key not in cfg:
            cfg[key] = extra[key]
    return cfg


def run_bertopic_phase(
    *,
    out: Path,
    meta_t: pd.DataFrame,
    gating_adapted: pd.DataFrame,
    h_t: np.ndarray,
    h_t_adapted: np.ndarray,
    method_name: str,
    bertopic_cfg: Dict[str, Any],
    topics_export_cfg: Dict[str, Any],
    text_col_t: str,
    repo_anchor: Path,
    corpus_id: Optional[str],
    topic_embedding_mode: Optional[str],
    topic_alpha: Optional[float],
    run_bertopic_grid: bool,
    grid_macros: Optional[Sequence[str]],
    skip_compression_diagnostics: bool,
) -> Dict[str, Any]:
    """Phase BERTopic: compression, grid optionnelle, fit intra-macro."""
    bertopic_cfg = dict(bertopic_cfg)
    if bertopic_cfg.get("enabled", True) is False:
        return {}

    topic_emb_cfg = resolve_topic_embedding_cfg(
        bertopic_cfg,
        cli_mode=topic_embedding_mode,
        cli_alpha=topic_alpha,
    )
    compression_path = None
    grid_path = None
    diagnostics_cfg = dict(bertopic_cfg.get("diagnostics") or {})
    grid_cfg = dict(bertopic_cfg.get("grid_search") or {})

    if (
        not skip_compression_diagnostics
        and diagnostics_cfg.get("enabled", True)
        and diagnostics_cfg.get("compute_compression", True)
    ):
        comp_df = compute_macro_compression_diagnostics(
            h_t,
            h_t_adapted,
            gating_adapted["m_hat"].astype(str).tolist(),
        )
        compression_path = out / "macro_compression_diagnostics.csv"
        comp_df.to_csv(compression_path, index=False)

    do_grid = run_bertopic_grid or bool(grid_cfg.get("enabled", False))
    if do_grid:
        macros_grid = list(grid_macros) if grid_macros else list(grid_cfg.get("macros", ["A0", "A1"]))
        texts_all = meta_t[text_col_t].astype(str).tolist()
        run_macro_bertopic_grid_search(
            texts_all,
            h_t,
            h_t_adapted,
            gating_adapted["m_hat"].astype(str).tolist(),
            macros=macros_grid,
            grid_cfg=grid_cfg,
            output_dir=out,
            bertopic_cfg=bertopic_cfg,
            random_state=int(bertopic_cfg.get("random_state", 42)),
            anchor=repo_anchor if repo_anchor.is_dir() else None,
        )
        grid_path = out / "bertopic_grid_A0_A1.csv"

    top_k_words = int(topics_export_cfg.get("top_k_words", 12))
    top_k_sentences = int(topics_export_cfg.get("top_k_sentences", 5))
    themes_bertopic, assignments_df, bertopic_partial = fit_bertopic_per_macro(
        h_t,
        meta_t,
        gating_adapted,
        method=method_name,
        bertopic_cfg=bertopic_cfg,
        output_dir=out / "topics_bertopic",
        legacy_output_dir=out / "topics_bertopic",
        per_macro_output_root=out / "bertopic",
        run_output_root=out,
        sentence_col=text_col_t,
        top_k_words=top_k_words,
        top_k_sentences=top_k_sentences,
        repo_anchor=repo_anchor if repo_anchor.is_dir() else None,
        corpus_id=corpus_id,
        embeddings_initial=h_t,
        embeddings_adapted=h_t_adapted,
        topic_embedding_cfg=topic_emb_cfg,
    )

    summary_dir = out / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    if not themes_bertopic.empty:
        summarize_topics_by_macro(themes_bertopic).to_csv(
            summary_dir / "topics_summary.csv", index=False
        )
    macro_counts = bertopic_partial.get("macro_topic_counts", {})
    macro_stats = build_macro_topic_test_table(
        macro_counts,
        assignments_df,
        themes_bertopic,
    )
    macro_stats.to_csv(summary_dir / "macro_topic_stats.csv", index=False)

    return {
        "embedding_mode": topic_emb_cfg["mode"],
        "alpha": topic_emb_cfg["alpha"],
        "normalize": topic_emb_cfg["normalize"],
        "macro_topic_counts": macro_counts,
        "macro_topic_stats_path": str(summary_dir / "macro_topic_stats.csv"),
        "warnings": bertopic_partial.get("warnings", []),
        "compression_diagnostics_path": str(compression_path) if compression_path else None,
        "grid_search_path": str(grid_path) if grid_path else None,
    }
