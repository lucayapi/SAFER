"""SoftTriple loss (entraînement sur embeddings encodeur HF unifié)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from contrastive_methods.distance import (
    center_merge_l21_penalty,
    center_pairwise_penalty,
    embedding_to_center_scores,
    maybe_l2_normalize,
    normalize_distance_metric,
)

VALID_CENTER_REGULARIZATION_TYPES = frozenset({"none", "merge_l21", "diversity"})


class SoftTripleLoss(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        centers_per_class: int = 5,
        gamma: float = 0.1,
        la: float = 10.0,
        delta: float = 0.01,
        tau: float = 0.0,
        normalize_embeddings: bool = True,
        normalize_centers: bool = True,
        center_max_similarity: float = 0.50,
        center_min_distance: float = 0.30,
        distance_metric: str = "euclidean",
        center_regularization_type: str = "none",
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.centers_per_class = int(centers_per_class)
        self.gamma = float(gamma)
        self.la = float(la)
        self.delta = float(delta)
        self.tau = float(tau)
        reg_type = str(center_regularization_type).strip().lower()
        if reg_type not in VALID_CENTER_REGULARIZATION_TYPES:
            raise ValueError(
                f"center_regularization_type invalide : {center_regularization_type!r} "
                f"(attendu : {sorted(VALID_CENTER_REGULARIZATION_TYPES)})"
            )
        self.center_regularization_type = reg_type
        self.distance_metric = normalize_distance_metric(distance_metric)
        use_cosine = self.distance_metric == "cosine"
        self.normalize_embeddings = bool(normalize_embeddings) and use_cosine
        self.normalize_centers = bool(normalize_centers) and use_cosine
        self.center_max_similarity = float(center_max_similarity)
        self.center_min_distance = float(center_min_distance)
        centers = torch.randn(num_classes, centers_per_class, embedding_dim) * 0.02
        if self.normalize_centers:
            centers = F.normalize(centers, p=2, dim=-1)
        self.centers = nn.Parameter(centers)

    def _get_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return maybe_l2_normalize(embeddings, self.normalize_embeddings)

    def _get_centers(self) -> torch.Tensor:
        return maybe_l2_normalize(self.centers, self.normalize_centers)

    def compute_relaxed_class_similarity(
        self, embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self._get_embeddings(embeddings)
        centers = self._get_centers()
        raw_sim = embedding_to_center_scores(z, centers, metric=self.distance_metric)
        if self.centers_per_class == 1:
            relaxed_sim = raw_sim.squeeze(-1)
        else:
            q = F.softmax(raw_sim / max(self.gamma, 1e-8), dim=2)
            relaxed_sim = (q * raw_sim).sum(dim=2)
        return relaxed_sim, raw_sim

    def regularization(self) -> torch.Tensor:
        if self.tau <= 0.0 or self.centers_per_class <= 1:
            return torch.tensor(0.0, device=self.centers.device)
        if self.center_regularization_type == "none":
            return torch.tensor(0.0, device=self.centers.device)
        centers = self._get_centers()
        penalties = []
        for c in range(self.num_classes):
            if self.center_regularization_type == "diversity":
                penalty = center_pairwise_penalty(
                    centers[c],
                    metric=self.distance_metric,
                    center_max_similarity=self.center_max_similarity,
                    center_min_distance=self.center_min_distance,
                )
            else:
                penalty = center_merge_l21_penalty(
                    centers[c],
                    metric=self.distance_metric,
                )
            if penalty.numel() > 0:
                penalties.append(penalty)
        if not penalties:
            return torch.tensor(0.0, device=self.centers.device)
        penalty = torch.stack(penalties).mean()
        if self.center_regularization_type == "merge_l21":
            penalty = penalty / (self.centers_per_class * (self.centers_per_class - 1))
        return self.tau * penalty

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        relaxed_sim, _ = self.compute_relaxed_class_similarity(embeddings)
        logits = self.la * relaxed_sim
        batch_idx = torch.arange(labels.shape[0], device=labels.device)
        logits = logits.clone()
        logits[batch_idx, labels] = self.la * (
            relaxed_sim[batch_idx, labels] - self.delta
        )
        ce = F.cross_entropy(logits, labels)
        reg = self.regularization()
        loss = ce + reg
        return loss, {
            "loss_total": float(loss.detach().cpu().item()),
            "loss_ce": float(ce.detach().cpu().item()),
            "loss_reg": float(reg.detach().cpu().item()),
        }


def make_collate_fn(tokenizer, max_length: int):
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [x["text"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

    return collate
