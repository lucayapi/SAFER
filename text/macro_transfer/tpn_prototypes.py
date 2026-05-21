"""Prototypes et responsabilités macro type TPN (numpy / torch)."""

from __future__ import annotations

import logging
from typing import Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

from macro_transfer.constants import LABEL2ID, MACRO_NAMES

logger = logging.getLogger(__name__)

DistanceMetric = Literal["euclidean", "cosine"]
AssignmentMode = Literal["soft", "hard"]

EPS = 1e-8


def l2_normalize_np(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        n = np.linalg.norm(x) + eps
        return x / n
    norms = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / norms


def l2_normalize_torch(x):
    import torch
    import torch.nn.functional as F

    return F.normalize(x, p=2, dim=-1, eps=EPS)


def pairwise_distances(
    x: np.ndarray,
    protos: np.ndarray,
    *,
    metric: DistanceMetric = "euclidean",
) -> np.ndarray:
    """
    x : (N, D), protos : (M, D), normalisés.
    Retourne (N, M) distances (euclidean squared ou 1 - cosine).
    """
    x = np.asarray(x, dtype=np.float64)
    protos = np.asarray(protos, dtype=np.float64)
    if metric == "cosine":
        sim = x @ protos.T
        return 1.0 - sim
    diff = x[:, None, :] - protos[None, :, :]
    return np.sum(diff * diff, axis=2)


def scores_from_prototypes(
    x: np.ndarray,
    protos: np.ndarray,
    *,
    tau: float = 0.3,
    metric: DistanceMetric = "euclidean",
) -> np.ndarray:
    """Logits = -d / tau, shape (N, M)."""
    d = pairwise_distances(x, protos, metric=metric)
    return -d / max(float(tau), EPS)


def soft_assignments(
    scores: np.ndarray,
    *,
    assignment_mode: AssignmentMode = "soft",
) -> np.ndarray:
    """Softmax sur scores ; hard → one-hot argmax."""
    s = np.asarray(scores, dtype=np.float64)
    s = s - s.max(axis=1, keepdims=True)
    exp_s = np.exp(s)
    q = exp_s / (exp_s.sum(axis=1, keepdims=True) + EPS)
    if assignment_mode == "hard":
        idx = np.argmax(q, axis=1)
        hard = np.zeros_like(q)
        hard[np.arange(len(q)), idx] = 1.0
        return hard
    return q


def macro_probs_from_source_prototypes(
    x: np.ndarray,
    mu_s: np.ndarray,
    *,
    tau: float = 0.3,
    metric: DistanceMetric = "euclidean",
    assignment_mode: AssignmentMode = "soft",
) -> np.ndarray:
    scores = scores_from_prototypes(x, mu_s, tau=tau, metric=metric)
    return soft_assignments(scores, assignment_mode=assignment_mode)


def compute_source_prototypes(
    h: np.ndarray,
    labels: Sequence[str],
    macros: Sequence[str] = MACRO_NAMES,
) -> np.ndarray:
    """μ_m^s = mean des h~ avec y=m, puis L2-normalise."""
    h = np.asarray(h, dtype=np.float64)
    labels = np.asarray(labels)
    protos = np.zeros((len(macros), h.shape[1]), dtype=np.float64)
    for mi, m in enumerate(macros):
        mask = labels == m
        if not mask.any():
            logger.warning("Aucun exemple source pour la macro %s", m)
            protos[mi] = 0.0
            continue
        protos[mi] = h[mask].mean(axis=0)
    return l2_normalize_np(protos)


def compute_target_prototypes_soft(
    h: np.ndarray,
    q: np.ndarray,
    *,
    eps: float = EPS,
) -> np.ndarray:
    """μ_m^t = sum_j q_jm h_j / (sum_j q_jm + eps), puis L2."""
    h = np.asarray(h, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    n_macros = q.shape[1]
    dim = h.shape[1]
    protos = np.zeros((n_macros, dim), dtype=np.float64)
    for mi in range(n_macros):
        mass = q[:, mi].sum()
        if mass < eps:
            logger.warning(
                "Masse cible quasi nulle pour macro index %d (mass=%.2e)",
                mi,
                mass,
            )
        w = q[:, mi : mi + 1]
        protos[mi] = (w * h).sum(axis=0) / (mass + eps)
    return l2_normalize_np(protos)


def compute_source_target_prototypes(
    h_s: np.ndarray,
    labels: Sequence[str],
    h_t: np.ndarray,
    q: np.ndarray,
    *,
    rho: float = 1.0,
    eps: float = EPS,
    macros: Sequence[str] = MACRO_NAMES,
) -> np.ndarray:
    """
    μ_m^st = (N_m^s * μ_s + ρ * sum_j q_jm h_j) / (N_m^s + ρ * sum_j q_jm + eps)
    avec μ_s = moyenne des h_s labellisés (non normalisée avant mix).
    """
    h_s = np.asarray(h_s, dtype=np.float64)
    h_t = np.asarray(h_t, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    labels = np.asarray(labels)
    dim = h_s.shape[1]
    protos = np.zeros((len(macros), dim), dtype=np.float64)
    for mi, m in enumerate(macros):
        src_mask = labels == m
        n_s = int(src_mask.sum())
        sum_src = h_s[src_mask].sum(axis=0) if n_s > 0 else np.zeros(dim, dtype=np.float64)
        mass_t = q[:, mi].sum()
        sum_tgt = (q[:, mi : mi + 1] * h_t).sum(axis=0)
        denom = n_s + rho * mass_t + eps
        protos[mi] = (sum_src + rho * sum_tgt) / denom
    return l2_normalize_np(protos)


def symmetric_kl(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> np.ndarray:
    """SKL(P,Q) par paire de lignes ou scalaire si 1D."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), eps, 1.0)
    if p.ndim == 1:
        kl_pq = np.sum(p * (np.log(p) - np.log(q)))
        kl_qp = np.sum(q * (np.log(q) - np.log(p)))
        return 0.5 * (kl_pq + kl_qp)
    kl_pq = np.sum(p * (np.log(p) - np.log(q)), axis=1)
    kl_qp = np.sum(q * (np.log(q) - np.log(p)), axis=1)
    return 0.5 * (kl_pq + kl_qp)


def distribution_from_prototypes_torch(
    h,
    mu,
    *,
    tau: float,
    metric: DistanceMetric,
):
    """P(m|x) softmax(-d/tau) pour batch torch."""
    import torch
    import torch.nn.functional as F

    if metric == "cosine":
        sim = h @ mu.T
        d = 1.0 - sim
    else:
        diff = h.unsqueeze(1) - mu.unsqueeze(0)
        d = (diff * diff).sum(dim=2)
    logits = -d / max(float(tau), EPS)
    return F.softmax(logits, dim=1)


def prototype_distance_table(
    mu_s: np.ndarray,
    mu_t: np.ndarray,
    mu_st: np.ndarray,
    macros: Sequence[str] = MACRO_NAMES,
) -> pd.DataFrame:
    """Distances euclidiennes au carré entre paires de prototypes."""
    rows = []
    pairs = [
        ("mu_s", "mu_t", mu_s, mu_t),
        ("mu_s", "mu_st", mu_s, mu_st),
        ("mu_t", "mu_st", mu_t, mu_st),
    ]
    for name_a, name_b, a, b in pairs:
        for mi, m in enumerate(macros):
            d = float(np.sum((a[mi] - b[mi]) ** 2))
            rows.append({"macro": m, "proto_a": name_a, "proto_b": name_b, "distance_sq": d})
    return pd.DataFrame(rows)


def labels_to_ids(labels: Sequence[str]) -> np.ndarray:
    return np.array([LABEL2ID.get(str(y), -1) for y in labels], dtype=np.int64)
