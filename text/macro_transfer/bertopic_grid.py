"""Grid search BERTopic ciblée A0/A1."""

from __future__ import annotations

import itertools
import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from macro_transfer.bertopic_exports import compute_topic_stats, topic_diversity_from_ids
from macro_transfer.bertopic_utils import fit_bertopic_subset
from macro_transfer.constants import MACRO_NAMES
from macro_transfer.topic_embeddings import build_topic_embeddings

logger = logging.getLogger(__name__)


def score_topic_config(row: Dict[str, Any], macro: str) -> float:
    """Score simple pour sélectionner une config grid."""
    n_topics = int(row.get("n_topics", 0))
    noise = float(row.get("noise_rate", 0.0))
    largest = float(row.get("largest_topic_share", 0.0))
    diversity = float(row.get("topic_diversity", 0.0) or 0.0)

    if macro == "A0":
        target_min, target_max = 5, 25
    elif macro == "A1":
        target_min, target_max = 5, 30
    else:
        target_min, target_max = 3, 40

    penalty = 0.0
    if n_topics < target_min:
        penalty += (target_min - n_topics) * 0.5
    if n_topics > target_max:
        penalty += (n_topics - target_max) * 0.2

    return 2.0 * diversity - 1.5 * noise - 1.0 * largest - penalty


def run_macro_bertopic_grid_search(
    docs: Sequence[str],
    embeddings_initial: np.ndarray,
    embeddings_adapted: np.ndarray,
    macro_labels: Sequence[str],
    macros: Optional[Sequence[str]] = None,
    grid_cfg: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
    *,
    bertopic_cfg: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
    anchor: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Grid cartésien alpha × min_cluster_size × min_samples × cluster_selection_method × macro.
    """
    grid_cfg = dict(grid_cfg or {})
    macros = list(macros) if macros is not None else list(grid_cfg.get("macros", ["A0", "A1"]))
    alphas = list(grid_cfg.get("alphas", [0.0, 0.25, 0.5, 0.75, 1.0]))
    min_cluster_sizes = list(grid_cfg.get("min_cluster_size", [10, 15, 20, 25, 30, 40]))
    min_samples_list = list(grid_cfg.get("min_samples", [1, 3, 5, 10]))
    csm_list = list(grid_cfg.get("cluster_selection_method", ["eom", "leaf"]))

    labels = np.asarray(macro_labels, dtype=object)
    emb_init = np.asarray(embeddings_initial, dtype=np.float64)
    emb_adapt = np.asarray(embeddings_adapted, dtype=np.float64)
    docs_list = list(docs)

    rows: list[dict] = []
    base_cfg = deepcopy(bertopic_cfg or {})
    rep = dict(base_cfg.get("representation") or {})
    rep["enabled"] = False
    base_cfg["representation"] = rep
    base_cfg["_disable_representation"] = True

    for macro in macros:
        mask = labels.astype(str) == str(macro)
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        texts_m = [docs_list[i] for i in idx]
        hi = emb_init[idx]
        ha = emb_adapt[idx]
        n_units = len(idx)

        for alpha, mcs, ms, csm in itertools.product(
            alphas, min_cluster_sizes, min_samples_list, csm_list
        ):
            mode = "initial" if float(alpha) <= 0.0 else ("adapted" if float(alpha) >= 1.0 else "mixed")
            t0 = time.perf_counter()
            try:
                emb_topic = build_topic_embeddings(
                    hi,
                    ha,
                    mode=mode,
                    alpha=float(alpha),
                    normalize=True,
                )
                cfg_run = deepcopy(base_cfg)
                macro_params = dict(cfg_run.get("macro_params") or {})
                mp = dict(macro_params.get(macro) or {})
                mp["min_cluster_size"] = int(mcs)
                mp["min_samples"] = int(ms)
                mp["cluster_selection_method"] = str(csm)
                mp["min_topic_size"] = int(mcs)
                macro_params[macro] = mp
                cfg_run["macro_params"] = macro_params

                topic_ids, _conf, _model = fit_bertopic_subset(
                    texts_m,
                    emb_topic,
                    cfg_run,
                    random_state=random_state,
                    anchor=anchor,
                    macro=macro,
                )
                stats = compute_topic_stats(topic_ids, n_units=n_units)
                diversity = topic_diversity_from_ids(topic_ids)
                runtime = time.perf_counter() - t0
                rows.append(
                    {
                        "macro": macro,
                        "alpha": float(alpha),
                        "embedding_mode": mode,
                        "min_cluster_size": int(mcs),
                        "min_samples": int(ms),
                        "cluster_selection_method": str(csm),
                        "n_units": n_units,
                        "n_topics": stats["n_topics"],
                        "n_noise": stats["n_noise"],
                        "noise_rate": stats["noise_rate"],
                        "largest_topic_size": stats["largest_topic_size"],
                        "largest_topic_share": stats["largest_topic_share"],
                        "median_topic_size": stats["median_topic_size"],
                        "mean_topic_size": stats["mean_topic_size"],
                        "empty_topics": stats["empty_topics"],
                        "topic_diversity": diversity,
                        "coherence_score": None,
                        "runtime_seconds": runtime,
                        "score": score_topic_config({**stats, "topic_diversity": diversity}, macro),
                    }
                )
            except Exception as exc:
                logger.warning(
                    "grid skip macro=%s alpha=%s mcs=%s ms=%s csm=%s: %s",
                    macro,
                    alpha,
                    mcs,
                    ms,
                    csm,
                    exc,
                )
                rows.append(
                    {
                        "macro": macro,
                        "alpha": float(alpha),
                        "embedding_mode": mode,
                        "min_cluster_size": int(mcs),
                        "min_samples": int(ms),
                        "cluster_selection_method": str(csm),
                        "n_units": n_units,
                        "n_topics": 0,
                        "n_noise": n_units,
                        "noise_rate": 1.0,
                        "largest_topic_size": 0,
                        "largest_topic_share": 1.0,
                        "median_topic_size": 0.0,
                        "mean_topic_size": 0.0,
                        "empty_topics": 0,
                        "topic_diversity": 0.0,
                        "coherence_score": None,
                        "runtime_seconds": time.perf_counter() - t0,
                        "score": -999.0,
                        "error": str(exc),
                    }
                )

    df = pd.DataFrame(rows)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "bertopic_grid_A0_A1.csv"
        df.to_csv(csv_path, index=False)
        if len(df):
            best_rows = []
            for macro in macros:
                sub = df[df["macro"].astype(str) == str(macro)]
                if sub.empty:
                    continue
                best = sub.loc[sub["score"].idxmax()]
                best_rows.append(best)
            if best_rows:
                pd.DataFrame(best_rows).to_csv(
                    output_dir / "bertopic_grid_A0_A1_best.csv", index=False
                )
    return df
