"""K-fold groupé (accident_id) et agrégation μ±σ des métriques géométriques."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

GEOMETRY_METRIC_KEYS: Tuple[str, ...] = (
    "delta_macro_pct",
    "eta2_macro_balanced",
    "eta2_weighted",
    "rankme_global",
    "c1_global",
    "c10_global",
)


def group_kfold_splits(
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Retourne une liste de (train_idx, val_idx) pour GroupKFold.

    GroupKFold (sklearn) n'accepte pas shuffle/random_state : on permute l'ordre
    des groupes avec ``seed`` puis on réattribue des ids entiers pour des folds reproductibles.
    """
    groups = np.asarray(groups).astype(str)
    unique_groups = np.unique(groups)
    n_splits = min(int(n_splits), len(unique_groups))
    if n_splits < 2:
        raise ValueError(f"n_splits doit être >= 2 (unique groups={len(unique_groups)}).")

    perm = unique_groups.copy()
    rng = np.random.RandomState(int(seed))
    rng.shuffle(perm)
    remap = {g: i for i, g in enumerate(perm)}
    group_ids = np.array([remap[g] for g in groups], dtype=np.int64)

    splitter = GroupKFold(n_splits=n_splits)
    indices = np.arange(len(groups))
    return [
        (train_idx.astype(np.int64), val_idx.astype(np.int64))
        for train_idx, val_idx in splitter.split(indices, groups=group_ids)
    ]


def aggregate_fold_rows(
    fold_rows: List[Dict[str, Any]],
    *,
    metric_keys: Sequence[str] = GEOMETRY_METRIC_KEYS,
) -> Dict[str, Any]:
    """Agrège des lignes par fold → mean/std + selection_score (mean delta_macro_pct)."""
    if not fold_rows:
        return {}
    df = pd.DataFrame(fold_rows)
    out: Dict[str, Any] = {"n_folds": len(df)}
    for key in metric_keys:
        if key not in df.columns:
            continue
        vals = pd.to_numeric(df[key], errors="coerce").dropna()
        if len(vals) == 0:
            out[f"mean_{key}"] = float("nan")
            out[f"std_{key}"] = float("nan")
        else:
            out[f"mean_{key}"] = float(vals.mean())
            out[f"std_{key}"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    if "mean_delta_macro_pct" in out:
        out["selection_score"] = out["mean_delta_macro_pct"]
    return out


def extract_test_metric_rows(fold_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extrait les métriques test (préfixe test_) pour agrégation kfold_test."""
    out: List[Dict[str, Any]] = []
    for row in fold_rows:
        entry: Dict[str, Any] = {"fold_id": row.get("fold_id")}
        for key in GEOMETRY_METRIC_KEYS:
            test_key = f"test_{key}"
            if test_key in row:
                entry[key] = row[test_key]
        if len(entry) > 1:
            out.append(entry)
    return out


def save_kfold_tables(
    fold_rows: List[Dict[str, Any]],
    metrics_dir,
    *,
    prefix: str = "kfold",
) -> Tuple[Any, Any]:
    from pathlib import Path

    from safer_core.io import ensure_dir

    metrics_dir = Path(metrics_dir)
    ensure_dir(metrics_dir)
    per_fold = pd.DataFrame(fold_rows)
    per_fold_path = metrics_dir / f"{prefix}_per_fold.csv"
    per_fold.to_csv(per_fold_path, index=False)
    summary = aggregate_fold_rows(fold_rows)
    summary_df = pd.DataFrame([summary])
    summary_path = metrics_dir / f"{prefix}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    return per_fold_path, summary_path
