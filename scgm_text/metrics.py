from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[Iterable[int]] = None,
) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)


def rankme_effective_rank(x: np.ndarray) -> float:
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        return float("nan")
    centered = x - x.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    total = singular_values.sum()
    if total <= 0:
        return float("nan")
    probs = singular_values / total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    return float(np.exp(entropy))


def pca_energy_c1_c10(x: np.ndarray) -> Tuple[float, float]:
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        return float("nan"), float("nan")
    centered = x - x.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    energy = singular_values ** 2
    total = energy.sum()
    if total <= 0:
        return float("nan"), float("nan")
    cumulative = np.cumsum(energy) / total
    c1 = float(cumulative[0])
    c10 = float(cumulative[min(9, len(cumulative) - 1)])
    return c1, c10


def rankme_by_macro(x: np.ndarray, y: np.ndarray) -> Dict[int, float]:
    results: Dict[int, float] = {}
    for label in np.unique(y):
        subset = x[y == label]
        results[int(label)] = rankme_effective_rank(subset)
    return results


def c1_c10_by_macro(x: np.ndarray, y: np.ndarray) -> Dict[int, Tuple[float, float]]:
    results: Dict[int, Tuple[float, float]] = {}
    for label in np.unique(y):
        subset = x[y == label]
        results[int(label)] = pca_energy_c1_c10(subset)
    return results


def silhouette_score_safe(x: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or len(labels) < 3:
        return float("nan")
    counts = np.bincount(labels.astype(np.int64), minlength=int(unique.max()) + 1)
    if np.any(counts[np.unique(labels)] < 2):
        return float("nan")
    try:
        return float(silhouette_score(x, labels))
    except ValueError:
        return float("nan")


def davies_bouldin_score_safe(x: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2:
        return float("nan")
    try:
        return float(davies_bouldin_score(x, labels))
    except ValueError:
        return float("nan")


def calinski_harabasz_score_safe(x: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2:
        return float("nan")
    try:
        return float(calinski_harabasz_score(x, labels))
    except ValueError:
        return float("nan")


def subtype_alignment_diagnostics(
    z_hat: np.ndarray,
    pred_subtype: np.ndarray,
) -> Dict[str, float]:
    mask = pred_subtype.astype(str) != "nan"
    if mask.sum() < 2:
        return {"nmi_subtype": float("nan"), "ari_subtype": float("nan")}
    labels = z_hat[mask]
    subtypes = pred_subtype[mask]
    if len(np.unique(labels)) < 2 or len(np.unique(subtypes)) < 2:
        return {"nmi_subtype": float("nan"), "ari_subtype": float("nan")}
    return {
        "nmi_subtype": float(normalized_mutual_info_score(subtypes, labels, average_method="arithmetic")),
        "ari_subtype": float(adjusted_rand_score(subtypes, labels)),
    }
