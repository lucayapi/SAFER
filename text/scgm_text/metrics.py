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


def mean_entropy(prob: np.ndarray, axis: int = -1) -> float:
    p = np.clip(prob, 1e-12, None)
    p = p / p.sum(axis=axis, keepdims=True)
    ent = -np.sum(p * np.log(p), axis=axis)
    return float(np.mean(ent))


def count_active_clusters(labels: np.ndarray) -> int:
    return int(np.unique(labels).size)


def q_assignment_distribution(q_hard: np.ndarray) -> Dict[str, float]:
    """Diagnostics from hard assignment matrix q (n x K, one-hot rows)."""
    z_hat = q_hard.argmax(axis=1)
    counts = np.bincount(z_hat, minlength=q_hard.shape[1])
    probs = counts / max(counts.sum(), 1)
    probs = probs[probs > 0]
    ent = -np.sum(probs * np.log(probs)) if probs.size else 0.0
    return {
        "n_active_z": float(count_active_clusters(z_hat)),
        "z_usage_entropy": float(ent),
        "z_max_mass": float(counts.max() / max(counts.sum(), 1)),
    }


def _label_valid_mask(labels: np.ndarray) -> np.ndarray:
    """True for usable cluster labels (numeric or string/object)."""
    arr = np.asarray(labels)
    if arr.size == 0:
        return np.zeros(0, dtype=bool)
    if np.issubdtype(arr.dtype, np.floating):
        return np.isfinite(arr)
    if np.issubdtype(arr.dtype, np.integer):
        return np.ones(arr.shape, dtype=bool)
    flat = arr.ravel()
    valid = np.empty(flat.shape[0], dtype=bool)
    for i, value in enumerate(flat):
        if value is None:
            valid[i] = False
            continue
        if isinstance(value, float) and not np.isfinite(value):
            valid[i] = False
            continue
        text = str(value).strip().lower()
        valid[i] = text not in ("", "nan", "none")
    return valid.reshape(arr.shape)


def homogeneity_purity_safe(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import homogeneity_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    mask = _label_valid_mask(y_true) & _label_valid_mask(y_pred)
    if mask.sum() < 2:
        return {"homogeneity_subtype": float("nan"), "purity_subtype": float("nan")}
    yt = y_true[mask].astype(str)
    yp = y_pred[mask].astype(str)
    if len(np.unique(yt)) < 2 or len(np.unique(yp)) < 2:
        return {"homogeneity_subtype": float("nan"), "purity_subtype": float("nan")}
    hom = float(homogeneity_score(yt, yp))
    # purity: max intersection / cluster size
    purity_vals = []
    for cluster in np.unique(yp):
        members = yt[yp == cluster]
        if members.size == 0:
            continue
        _, counts = np.unique(members, return_counts=True)
        purity_vals.append(counts.max() / members.size)
    purity = float(np.mean(purity_vals)) if purity_vals else float("nan")
    return {"homogeneity_subtype": hom, "purity_subtype": purity}
