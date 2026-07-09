"""Batch Hard Triplet sur embeddings (encodeur HF unifié)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.triplet_diagnostics import (
    TripletDiagnosticLogger,
    compute_batch_hard_triplet_stats,
    raise_on_invalid_triplet_batch,
    triplet_loss_from_hard_distances,
)


def _resolve_soft_margin(cfg: ContrastiveConfig) -> bool:
    loss_type = (cfg.triplet_loss_type or "soft_margin").strip().lower()
    if loss_type in ("hard", "batch_hard"):
        return False
    if loss_type in ("soft_margin", "soft", "batch_hard_soft_margin"):
        return True
    if cfg.batch_triplet_margin is not None:
        return False
    return True


class BatchTripletEmbeddingLoss(nn.Module):
    """Batch-hard triplet sur embeddings [B, D]."""

    def __init__(
        self,
        *,
        distance_metric: str,
        soft_margin: bool = True,
        margin: Optional[float] = None,
        diagnostics_eps: float = 1e-6,
        diagnostic_logger: Optional[TripletDiagnosticLogger] = None,
    ) -> None:
        super().__init__()
        self.distance_metric = distance_metric
        self.soft_margin = bool(soft_margin)
        self.margin = margin
        self.diagnostics_eps = float(diagnostics_eps)
        self.diagnostic_logger = diagnostic_logger

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.contiguous().view(-1).long()
        stats = compute_batch_hard_triplet_stats(
            embeddings,
            labels,
            distance_metric=self.distance_metric,
            soft_margin=self.soft_margin,
            margin=self.margin,
            eps=self.diagnostics_eps,
        )
        if stats.n_valid_anchors > 0:
            valid = stats.valid_mask
            loss = triplet_loss_from_hard_distances(
                stats.d_pos[valid],
                stats.d_neg[valid],
                soft_margin=self.soft_margin,
                margin=self.margin,
            )
        else:
            raise_on_invalid_triplet_batch(stats)
        if self.diagnostic_logger is not None:
            with torch.no_grad():
                self.diagnostic_logger.maybe_log(stats, float(loss.detach().item()))
        return loss


def build_batch_triplet_embedding_loss(
    cfg: ContrastiveConfig,
    *,
    diagnostic_logger: Optional[TripletDiagnosticLogger] = None,
) -> BatchTripletEmbeddingLoss:
    soft_margin = _resolve_soft_margin(cfg)
    margin = cfg.batch_triplet_margin if not soft_margin else None
    return BatchTripletEmbeddingLoss(
        distance_metric=cfg.distance_metric,
        soft_margin=soft_margin,
        margin=margin,
        diagnostics_eps=cfg.triplet_diagnostics_eps,
        diagnostic_logger=diagnostic_logger,
    )
