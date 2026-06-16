"""Prototypes macro et opérations d'assignation (numpy / torch)."""

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


def pairwise_distances_torch(h, protos, *, metric: DistanceMetric = "euclidean"):
    """h (N,D), protos (M,D) → distances (N,M)."""
    import torch

    if metric == "cosine":
        sim = h @ protos.T
        return 1.0 - sim
    diff = h.unsqueeze(1) - protos.unsqueeze(0)
    return (diff * diff).sum(dim=2)


def prototype_logits_torch(
    h,
    protos,
    *,
    tau: float = 0.3,
    metric: DistanceMetric = "euclidean",
):
    """Logits = -d/tau, shape (N, M)."""
    d = pairwise_distances_torch(h, protos, metric=metric)
    return -d / max(float(tau), EPS)


def soft_assignments_torch(logits, *, assignment_mode: AssignmentMode = "soft"):
    """Softmax ou one-hot argmax sur logits (N, M)."""
    import torch
    import torch.nn.functional as F

    if assignment_mode == "hard":
        idx = logits.argmax(dim=1)
        q = F.one_hot(idx, num_classes=logits.shape[1]).to(dtype=logits.dtype)
        return q
    return F.softmax(logits, dim=1)


def distribution_from_prototypes_torch(
    h,
    mu,
    *,
    tau: float,
    metric: DistanceMetric,
):
    """P(m|x) = softmax(-d/tau) pour batch torch."""
    import torch.nn.functional as F

    logits = prototype_logits_torch(h, mu, tau=tau, metric=metric)
    return F.softmax(logits, dim=1)


def prototype_distance_torch(a, b, *, metric: DistanceMetric = "euclidean"):
    """Distance scalaire entre deux prototypes (D,) ou (1,D)."""
    import torch

    a = a.view(-1)
    b = b.view(-1)
    if metric == "cosine":
        return 1.0 - (a * b).sum()
    return ((a - b) ** 2).sum()


def compute_source_prototypes_torch(
    htilde_s,
    y_ids,
    n_macros: int,
    *,
    eps: float = EPS,
):
    """μ_m^s + validité et counts source (différentiable)."""
    import torch
    import torch.nn.functional as F

    dim = htilde_s.shape[1]
    device = htilde_s.device
    dtype = htilde_s.dtype
    protos = torch.zeros((n_macros, dim), device=device, dtype=dtype)
    valid = torch.zeros((n_macros,), device=device, dtype=torch.bool)
    counts = torch.zeros((n_macros,), device=device, dtype=torch.long)
    for m in range(n_macros):
        mask = y_ids == m
        counts[m] = mask.sum()
        if not mask.any():
            continue
        valid[m] = True
        protos[m] = htilde_s[mask].mean(dim=0)
    return F.normalize(protos, p=2, dim=-1, eps=eps), valid, counts


def compute_target_prototypes_soft_torch(htilde_t, q, *, eps: float = EPS):
    """μ_m^t + validité et masses cibles."""
    import torch
    import torch.nn.functional as F

    n_macros = q.shape[1]
    dim = htilde_t.shape[1]
    device = htilde_t.device
    dtype = htilde_t.dtype
    protos = torch.zeros((n_macros, dim), device=device, dtype=dtype)
    valid = torch.zeros((n_macros,), device=device, dtype=torch.bool)
    masses = torch.zeros((n_macros,), device=device, dtype=dtype)
    for mi in range(n_macros):
        mass = q[:, mi].sum()
        masses[mi] = mass
        if mass < eps:
            continue
        valid[mi] = True
        w = q[:, mi : mi + 1]
        protos[mi] = (w * htilde_t).sum(dim=0) / (mass + eps)
    return F.normalize(protos, p=2, dim=-1, eps=eps), valid, masses


def compute_source_target_prototypes_torch(
    htilde_s,
    y_ids,
    htilde_t,
    q,
    n_macros: int,
    *,
    rho: float = 1.0,
    eps: float = EPS,
):
    """μ_m^st + validité et masses cible pour le terme source-target."""
    import torch
    import torch.nn.functional as F

    dim = htilde_s.shape[1]
    device = htilde_s.device
    dtype = htilde_s.dtype
    protos = torch.zeros((n_macros, dim), device=device, dtype=dtype)
    valid = torch.zeros((n_macros,), device=device, dtype=torch.bool)
    masses = torch.zeros((n_macros,), device=device, dtype=dtype)
    for m in range(n_macros):
        src_mask = y_ids == m
        n_s = int(src_mask.sum().item())
        sum_src = htilde_s[src_mask].sum(dim=0) if n_s > 0 else torch.zeros(dim, device=device, dtype=dtype)
        mass_t = q[:, m].sum()
        masses[m] = mass_t
        sum_tgt = (q[:, m : m + 1] * htilde_t).sum(dim=0)
        denom = n_s + rho * mass_t + eps
        protos[m] = (sum_src + rho * sum_tgt) / denom
        if (n_s > 0) or float(mass_t.item()) >= eps:
            valid[m] = True
    return F.normalize(protos, p=2, dim=-1, eps=eps), valid, masses


def symmetric_kl_torch(p, q, *, eps: float = EPS):
    """SKL(P,Q) par ligne, shape (N,) si batch."""
    import torch

    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    kl_pq = (p * (torch.log(p) - torch.log(q))).sum(dim=-1)
    kl_qp = (q * (torch.log(q) - torch.log(p))).sum(dim=-1)
    return 0.5 * (kl_pq + kl_qp)


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
