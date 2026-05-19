"""Macro geometry separation via eta-squared (balanced and weighted)."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from scgm_text.dataset_text_embeddings import ID2LABEL, LABEL2ID
from scgm_text.metrics import pca_energy_c1_c10, rankme_effective_rank

MACRO_NAMES: Tuple[str, ...] = ("A0", "A1", "B", "C")

PRIMARY_SELECTION_METRIC = "eta2_macro_balanced_perc"

GEOMETRY_METRIC_KEYS: Tuple[str, ...] = (
    "eta2_macro_balanced",
    "eta2_macro_balanced_perc",
    "eta2_weighted",
    "T_macro_balanced",
    "W_macro_balanced",
    "B_macro_balanced",
    "T_weighted",
    "W_weighted",
    "B_weighted",
    "W_A0",
    "n_A0",
    "W_A1",
    "n_A1",
    "W_B",
    "n_B",
    "W_C",
    "n_C",
    "rankme_global",
    "c1_global",
    "c10_global",
)

METRICS_TABLE_COLUMNS: List[str] = [
    "method",
    "eta2_macro_balanced",
    "eta2_macro_balanced_perc",
    "eta2_weighted",
    "T_macro_balanced",
    "W_macro_balanced",
    "B_macro_balanced",
    "T_weighted",
    "W_weighted",
    "B_weighted",
    "W_A0",
    "n_A0",
    "W_A1",
    "n_A1",
    "W_B",
    "n_B",
    "W_C",
    "n_C",
    "rankme_global",
    "c1_global",
    "c10_global",
    "macros_ignored",
]


def _as_numpy(x: Union[np.ndarray, Any]) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def _macro_labels_to_ids(macro_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(macro_labels)
    n = arr.shape[0]
    ids = np.full(n, -1, dtype=np.int64)
    if np.issubdtype(arr.dtype, np.integer) and not np.issubdtype(arr.dtype, np.bool_):
        for i, value in enumerate(arr):
            if 0 <= int(value) <= 3:
                ids[i] = int(value)
        return ids, ids >= 0
    for i, value in enumerate(arr):
        if value is None:
            continue
        if isinstance(value, float) and not np.isfinite(value):
            continue
        text = str(value).strip()
        if text in LABEL2ID:
            ids[i] = LABEL2ID[text]
    return ids, ids >= 0


def _intra_inertia(z: np.ndarray) -> float:
    """Mean squared distance to class centroid: (1/n_c) sum ||z_i - mu_c||^2."""
    n_c = z.shape[0]
    if n_c < 2:
        return float("nan")
    mu_c = z.mean(axis=0)
    diff = z - mu_c
    return float(np.mean(np.einsum("ij,ij->i", diff, diff)))


def _scatter_to_centroid(z: np.ndarray, centroid: np.ndarray) -> float:
    """(1/n) sum ||z_i - centroid||^2."""
    n = z.shape[0]
    if n == 0:
        return float("nan")
    diff = z - centroid
    return float(np.mean(np.einsum("ij,ij->i", diff, diff)))


def _eta2_from_tw(w: float, t: float, eps: float, metric_name: str) -> float:
    if not np.isfinite(t) or t <= eps:
        warnings.warn(
            f"{metric_name}: total inertia T={t} is near zero; eta2 set to NaN.",
            UserWarning,
            stacklevel=3,
        )
        return float("nan")
    eta2 = 1.0 - w / (t + eps)
    return float(np.clip(eta2, 0.0, 1.0))


def compute_eta2_macro_geometry(
    X: Union[np.ndarray, Any],
    y: np.ndarray,
    labels: Sequence[str] = MACRO_NAMES,
    l2_normalize: bool = False,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Eta-squared macro geometry on squared Euclidean distances (no classifier).

    eta2_macro_balanced is the primary score; eta2_weighted is secondary.
    """
    z = _as_numpy(X)
    label_ids, valid_mask = _macro_labels_to_ids(y)
    z = z[valid_mask]
    label_ids = label_ids[valid_mask]

    if l2_normalize:
        z = _l2_normalize_rows(z)

    nan_out: Dict[str, Any] = {
        "eta2_macro_balanced": float("nan"),
        "eta2_weighted": float("nan"),
        "T_macro_balanced": float("nan"),
        "W_macro_balanced": float("nan"),
        "B_macro_balanced": float("nan"),
        "T_weighted": float("nan"),
        "W_weighted": float("nan"),
        "B_weighted": float("nan"),
        "macros_valid": "",
        "macros_ignored": "",
    }
    for name in labels:
        nan_out[f"W_{name}"] = float("nan")
        nan_out[f"n_{name}"] = 0

    if z.shape[0] < 2:
        return nan_out

    w_by_macro: Dict[str, float] = {}
    n_by_macro: Dict[str, int] = {}
    macros_ignored: List[str] = []
    macros_valid: List[str] = []
    centroids: Dict[str, np.ndarray] = {}

    for name in labels:
        macro_id = LABEL2ID.get(name)
        if macro_id is None:
            continue
        mask = label_ids == macro_id
        n_c = int(mask.sum())
        n_by_macro[name] = n_c
        if n_c < 2:
            w_by_macro[name] = float("nan")
            if n_c == 1:
                macros_ignored.append(name)
            continue
        z_c = z[mask]
        w_by_macro[name] = _intra_inertia(z_c)
        centroids[name] = z_c.mean(axis=0)
        macros_valid.append(name)

    for name in labels:
        nan_out[f"W_{name}"] = w_by_macro.get(name, float("nan"))
        nan_out[f"n_{name}"] = n_by_macro.get(name, 0)

    nan_out["macros_ignored"] = ",".join(macros_ignored)
    nan_out["macros_valid"] = ",".join(macros_valid)

    if not macros_valid:
        return nan_out

    n_valid = int(z.shape[0])
    w_vals = [w_by_macro[m] for m in macros_valid]

    mu_balanced = np.mean([centroids[m] for m in macros_valid], axis=0)
    w_macro_balanced = float(np.mean(w_vals))
    t_parts = []
    for name in macros_valid:
        mask = label_ids == LABEL2ID[name]
        t_parts.append(_scatter_to_centroid(z[mask], mu_balanced))
    t_macro_balanced = float(np.mean(t_parts))
    b_macro_balanced = t_macro_balanced - w_macro_balanced

    mu_weighted = z.mean(axis=0)
    t_weighted = _scatter_to_centroid(z, mu_weighted)
    w_weighted = float(
        sum((n_by_macro[m] / n_valid) * w_by_macro[m] for m in macros_valid)
    )
    b_weighted = t_weighted - w_weighted

    nan_out.update(
        {
            "W_macro_balanced": w_macro_balanced,
            "T_macro_balanced": t_macro_balanced,
            "B_macro_balanced": b_macro_balanced,
            "eta2_macro_balanced": _eta2_from_tw(
                w_macro_balanced, t_macro_balanced, eps, "eta2_macro_balanced"
            ),
            "W_weighted": w_weighted,
            "T_weighted": t_weighted,
            "B_weighted": b_weighted,
            "eta2_weighted": _eta2_from_tw(w_weighted, t_weighted, eps, "eta2_weighted"),
        }
    )
    return nan_out


def build_geometry_metrics_row(
    X: Union[np.ndarray, Any],
    y: np.ndarray,
    *,
    method: str,
    labels: Sequence[str] = MACRO_NAMES,
    l2_normalize: bool = False,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Full metrics table row: eta2 geometry + RankMe / C1 / C10."""
    z = _as_numpy(X)
    label_ids, valid_mask = _macro_labels_to_ids(y)
    z_valid = z[valid_mask]
    if l2_normalize:
        z_valid = _l2_normalize_rows(z_valid)

    row = compute_eta2_macro_geometry(
        z,
        y,
        labels=labels,
        l2_normalize=l2_normalize,
        eps=eps,
    )
    row["method"] = method

    if z_valid.shape[0] >= 2:
        c1, c10 = pca_energy_c1_c10(z_valid)
        row["rankme_global"] = rankme_effective_rank(z_valid)
        row["c1_global"] = c1
        row["c10_global"] = c10
    else:
        row["rankme_global"] = float("nan")
        row["c1_global"] = float("nan")
        row["c10_global"] = float("nan")

    eta2 = row.get("eta2_macro_balanced", float("nan"))
    if np.isfinite(eta2):
        row["eta2_macro_balanced_perc"] = float(100.0 * eta2)
    else:
        row["eta2_macro_balanced_perc"] = float("nan")

    ordered: Dict[str, Any] = {"method": method}
    for col in METRICS_TABLE_COLUMNS:
        if col == "method":
            continue
        if col in row:
            ordered[col] = row[col]
        elif col.startswith("n_"):
            ordered[col] = 0
        else:
            ordered[col] = float("nan")
    return ordered


def metrics_table_from_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize rows to METRICS_TABLE_COLUMNS order."""
    normalized = []
    for row in rows:
        normalized.append({col: row.get(col, np.nan) for col in METRICS_TABLE_COLUMNS})
    return normalized
