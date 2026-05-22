"""Diagnostics de compression intra-macro (embeddings init vs adaptés)."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from macro_transfer.constants import MACRO_NAMES

EPS = 1e-8


def _within_macro_variance(h: np.ndarray) -> float:
    """W = mean_j ||h_j - mean(h)||^2."""
    h = np.asarray(h, dtype=np.float64)
    if len(h) == 0:
        return float("nan")
    centroid = h.mean(axis=0)
    return float(np.mean(np.sum((h - centroid) ** 2, axis=1)))


def _mean_knn_distance(h: np.ndarray, k: int = 10, max_samples: int = 2000) -> float:
    """Distance moyenne au k-ième voisin (euclidien), sous-échantillon si N grand."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    if n <= 1:
        return float("nan")
    k_eff = min(k, n - 1)
    if n > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_samples, replace=False)
        h = h[idx]
        n = len(h)
    dists = np.sqrt(np.maximum(0.0, np.sum((h[:, None, :] - h[None, :, :]) ** 2, axis=2)))
    np.fill_diagonal(dists, np.inf)
    knn = np.partition(dists, k_eff, axis=1)[:, k_eff]
    return float(np.mean(knn))


def compute_macro_compression_diagnostics(
    h_initial: np.ndarray,
    h_adapted: np.ndarray,
    macro_labels: Sequence[str],
    macros: Optional[Sequence[str]] = None,
    *,
    k_neighbors: int = 10,
) -> pd.DataFrame:
    """
    Pour chaque macro m :
      W_init, W_adapt, compression_ratio = W_adapt / (W_init + eps)
    Optionnel : mean_knn_distance_init, mean_knn_distance_adapt.
    """
    h_initial = np.asarray(h_initial, dtype=np.float64)
    h_adapted = np.asarray(h_adapted, dtype=np.float64)
    if h_initial.shape != h_adapted.shape:
        raise ValueError(
            f"h_initial {h_initial.shape} et h_adapted {h_adapted.shape} incompatibles"
        )
    labels = np.asarray(macro_labels, dtype=object)
    macro_list = list(macros) if macros is not None else list(MACRO_NAMES)

    rows: list[dict[str, Any]] = []
    for macro in macro_list:
        mask = labels.astype(str) == str(macro)
        n_units = int(mask.sum())
        if n_units == 0:
            rows.append(
                {
                    "macro": macro,
                    "n_units": 0,
                    "W_init": float("nan"),
                    "W_adapt": float("nan"),
                    "compression_ratio": float("nan"),
                    "mean_knn_distance_init": float("nan"),
                    "mean_knn_distance_adapt": float("nan"),
                }
            )
            continue
        hi = h_initial[mask]
        ha = h_adapted[mask]
        w_init = _within_macro_variance(hi)
        w_adapt = _within_macro_variance(ha)
        ratio = w_adapt / (w_init + EPS)
        rows.append(
            {
                "macro": macro,
                "n_units": n_units,
                "W_init": w_init,
                "W_adapt": w_adapt,
                "compression_ratio": ratio,
                "mean_knn_distance_init": _mean_knn_distance(hi, k=k_neighbors),
                "mean_knn_distance_adapt": _mean_knn_distance(ha, k=k_neighbors),
            }
        )
    return pd.DataFrame(rows)
