"""
Supervised Contrastive Loss (HobbitLong / SupContrast).

Adapté de https://github.com/HobbitLong/SupContrast (BSD-2-Clause).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from contrastive_methods.config import ContrastiveConfig


class HobbitSupConLoss(nn.Module):
    """SupConLoss officiel (features [bsz, n_views, dim], L2-normalisées en amont)."""

    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = "all",
        base_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.contrast_mode = str(contrast_mode)
        self.base_temperature = float(base_temperature)

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = features.device

        if features.dim() < 3:
            raise ValueError(
                "`features` must be [bsz, n_views, dim], at least 3 dimensions"
            )
        if features.dim() > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        if labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown contrast_mode: {self.contrast_mode}")

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature,
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask
        valid_anchor = mask.sum(1) > 0

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size)
        valid_anchor = valid_anchor.view(anchor_count, batch_size)
        if valid_anchor.any():
            return loss[valid_anchor].mean()
        return loss.mean()


class SupConEmbeddingLoss(nn.Module):
    """SupCon sur embeddings [B, D] (encodeur HF unifié)."""

    def __init__(
        self,
        *,
        temperature: float = 0.07,
        contrast_mode: str = "all",
        base_temperature: Optional[float] = None,
        normalize_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.normalize_embeddings = bool(normalize_embeddings)
        base_t = float(base_temperature) if base_temperature is not None else float(temperature)
        self.criterion = HobbitSupConLoss(
            temperature=float(temperature),
            contrast_mode=str(contrast_mode),
            base_temperature=base_t,
        )

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        features = embeddings.unsqueeze(1)
        return self.criterion(features, labels)


def build_supcon_embedding_loss(cfg: ContrastiveConfig) -> SupConEmbeddingLoss:
    if cfg.method_name == "supcon" and cfg.distance_metric != "cosine":
        raise ValueError(
            f"SupCon (HobbitLong) exige distance_metric=cosine, reçu {cfg.distance_metric!r}"
        )
    return SupConEmbeddingLoss(
        temperature=cfg.supcon_temperature,
        contrast_mode=cfg.supcon_contrast_mode,
        base_temperature=cfg.supcon_base_temperature,
        normalize_embeddings=cfg.supcon_normalize_embeddings,
    )
