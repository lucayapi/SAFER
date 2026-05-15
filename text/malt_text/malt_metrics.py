from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from scgm_text.dataset_text_embeddings import ID2LABEL
from scgm_text.metrics import (
    accuracy,
    balanced_accuracy,
    c1_c10_by_macro,
    macro_f1,
    pca_energy_c1_c10,
    rankme_by_macro,
    rankme_effective_rank,
)


def entropy(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    clipped = np.clip(probs, 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=axis)


def anchor_drift_metrics(
    mu_source: np.ndarray,
    mu_target: np.ndarray,
) -> Dict[str, float]:
    source_norm = mu_source / np.linalg.norm(mu_source, axis=1, keepdims=True).clip(min=1e-12)
    target_norm = mu_target / np.linalg.norm(mu_target, axis=1, keepdims=True).clip(min=1e-12)
    cosine = np.sum(source_norm * target_norm, axis=1)
    l2 = np.linalg.norm(source_norm - target_norm, axis=1)
    metrics = {
        "anchor_drift_mean": float(l2.mean()),
    }
    for macro_id, macro_name in ID2LABEL.items():
        metrics[f"anchor_drift_{macro_name}"] = float(l2[macro_id])
        metrics[f"anchor_cosine_{macro_name}"] = float(cosine[macro_id])
    return metrics


def probability_summary(probs: np.ndarray) -> Dict[str, float]:
    return {
        "mean_entropy": float(entropy(probs).mean()),
        "mean_max": float(probs.max(axis=1).mean()),
    }


def active_cluster_count(z_hat: np.ndarray, k: int) -> int:
    counts = np.bincount(z_hat, minlength=k)
    return int(np.sum(counts > 0))


def cluster_entropy(z_hat: np.ndarray, k: int) -> float:
    counts = np.bincount(z_hat, minlength=k).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return float("nan")
    probs = counts / total
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def mean_py_given_z_entropy(prob_y_z: np.ndarray) -> float:
    return float(entropy(prob_y_z, axis=1).mean())


def diagnostic_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    valid = y_true >= 0
    if not np.any(valid):
        return {}
    return {
        "macro_f1": macro_f1(y_true[valid], y_pred[valid]),
        "balanced_accuracy": balanced_accuracy(y_true[valid], y_pred[valid]),
        "accuracy": accuracy(y_true[valid], y_pred[valid]),
    }


def geometry_metrics(
    projected: np.ndarray,
    macro_labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    c1, c10 = pca_energy_c1_c10(projected)
    metrics = {
        "rankme": rankme_effective_rank(projected),
        "c1": c1,
        "c10": c10,
    }
    if macro_labels is not None:
        valid = macro_labels >= 0
        if np.any(valid):
            rankme_macro = rankme_by_macro(projected[valid], macro_labels[valid])
            c1_macro = c1_c10_by_macro(projected[valid], macro_labels[valid])
            for macro_id, macro_name in ID2LABEL.items():
                metrics[f"rankme_{macro_name}"] = float(rankme_macro.get(macro_id, np.nan))
                c1_value, c10_value = c1_macro.get(macro_id, (np.nan, np.nan))
                metrics[f"c1_{macro_name}"] = float(c1_value)
                metrics[f"c10_{macro_name}"] = float(c10_value)
    return metrics
