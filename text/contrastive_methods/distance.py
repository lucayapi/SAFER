"""Métriques de distance partagées (SupCon, SoftTriple, triplet ST)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

VALID_DISTANCE_METRICS = frozenset({"euclidean", "cosine", "eucledian"})


def normalize_distance_metric(name: str) -> str:
    key = (name or "euclidean").strip().lower()
    if key == "eucledian":
        key = "euclidean"
    if key not in VALID_DISTANCE_METRICS:
        raise ValueError(
            f"distance_metric inconnue : {name!r} (attendu : 'euclidean' ou 'cosine')"
        )
    return key


def maybe_l2_normalize(embeddings: torch.Tensor, normalize: bool) -> torch.Tensor:
    if normalize:
        return F.normalize(embeddings, p=2, dim=-1)
    return embeddings


def pairwise_logits(
    embeddings: torch.Tensor,
    *,
    metric: str,
    temperature: float,
) -> torch.Tensor:
    """Logits in-batch pour SupCon : euclidien = -||zi-zj||²/τ ; cosinus = zi·zj/τ."""
    metric = normalize_distance_metric(metric)
    temp = max(float(temperature), 1e-8)
    if metric == "euclidean":
        dist_sq = torch.cdist(embeddings, embeddings, p=2).pow(2)
        return -dist_sq / temp
    return torch.matmul(embeddings, embeddings.T) / temp


def embedding_to_center_scores(
    embeddings: torch.Tensor,
    centers: torch.Tensor,
    *,
    metric: str,
) -> torch.Tensor:
    """
    Scores point-centre [B, C, K] (plus grand = plus proche du centre).
    Euclidien : -||z - c||² ; cosinus : produit scalaire (centres supposés L2-normés).
    """
    metric = normalize_distance_metric(metric)
    if metric == "euclidean":
        diff = embeddings.unsqueeze(1).unsqueeze(2) - centers.unsqueeze(0)
        return -(diff.pow(2).sum(dim=-1))
    return torch.einsum("bd,ckd->bck", embeddings, centers)


def center_pairwise_penalty(
    centers_class: torch.Tensor,
    *,
    metric: str,
    center_max_similarity: float,
    center_min_distance: float,
) -> torch.Tensor:
    """Régularisation intra-classe entre centroïdes SoftTriple."""
    metric = normalize_distance_metric(metric)
    k = centers_class.shape[0]
    if k <= 1:
        return torch.tensor(0.0, device=centers_class.device)
    iu = torch.triu_indices(k, k, offset=1, device=centers_class.device)
    if metric == "euclidean":
        dist = torch.cdist(centers_class, centers_class, p=2)
        pair_dist = dist[iu[0], iu[1]]
        margin = float(center_min_distance)
        return F.relu(margin - pair_dist).pow(2).mean()
    sim = torch.clamp(centers_class @ centers_class.T, -1.0, 1.0)
    pair_sim = sim[iu[0], iu[1]]
    return F.relu(pair_sim - float(center_max_similarity)).pow(2).mean()
