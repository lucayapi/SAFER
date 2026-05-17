"""Supervised Contrastive Loss (SentenceTransformer)."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(
        self,
        model: Any,
        temperature: float = 0.07,
        normalize_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.temperature = float(temperature)
        self.normalize_embeddings = bool(normalize_embeddings)

    def forward(self, sentence_features, labels: Optional[torch.Tensor] = None):
        if labels is None:
            raise ValueError("SupConLoss nécessite des labels.")

        embeddings = self.model(sentence_features[0])["sentence_embedding"]
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        labels = labels.view(-1)
        device = embeddings.device
        batch_size = embeddings.size(0)
        if batch_size < 2:
            return embeddings.sum() * 0.0

        logits = torch.matmul(embeddings, embeddings.T) / self.temperature
        eye_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        logits = logits.masked_fill(eye_mask, float("-inf"))

        labels = labels.contiguous().view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).to(device) & (~eye_mask)
        positives_per_anchor = positive_mask.sum(dim=1)
        valid_anchor_mask = positives_per_anchor > 0
        if not valid_anchor_mask.any():
            return embeddings.sum() * 0.0

        log_denom = torch.logsumexp(logits, dim=1, keepdim=True)
        log_prob = logits - log_denom
        mean_log_prob_pos = (
            (positive_mask.float() * log_prob).sum(dim=1)
            / positives_per_anchor.clamp(min=1).float()
        )
        return -mean_log_prob_pos[valid_anchor_mask].mean()
