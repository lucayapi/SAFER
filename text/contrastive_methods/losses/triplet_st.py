"""
Batch Hard Triplet (Sentence Transformers + diagnostics optionnels).

Utilise ``BatchHardSoftMarginTripletLoss`` ou un wrapper custom avec monitoring.
https://sbert.net/docs/sentence_transformer/loss_overview.html

Requiert ``BatchSamplers.GROUP_BY_LABEL`` (voir ``st_common.build_training_arguments``).
"""

from __future__ import annotations

import inspect
from typing import Any, Optional

import torch
import torch.nn as nn

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.st_common import resolve_triplet_distance
from contrastive_methods.triplet_diagnostics import (
    TripletDiagnosticLogger,
    compute_batch_hard_triplet_stats,
    triplet_loss_from_hard_distances,
)


class BatchTripletLossWithDiagnostics(nn.Module):
    """Encode → batch-hard triplet loss + diagnostics CSV/console."""

    def __init__(
        self,
        model: Any,
        *,
        distance_metric: str,
        soft_margin: bool = True,
        margin: Optional[float] = None,
        diagnostics_eps: float = 1e-6,
        diagnostic_logger: Optional[TripletDiagnosticLogger] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.distance_metric = distance_metric
        self.soft_margin = bool(soft_margin)
        self.margin = margin
        self.diagnostics_eps = float(diagnostics_eps)
        self.diagnostic_logger = diagnostic_logger
        self.loss_type = "soft_margin" if self.soft_margin else "hard"

    def set_training_context(
        self,
        *,
        global_step: Optional[int] = None,
        epoch: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ) -> None:
        if self.diagnostic_logger is None:
            return
        self.diagnostic_logger.set_training_context(
            global_step=global_step,
            epoch=epoch,
            learning_rate=learning_rate,
        )

    def forward(self, sentence_features, labels: Optional[torch.Tensor] = None):
        if labels is None:
            raise ValueError("BatchTripletLossWithDiagnostics nécessite des labels.")
        embeddings = self.model(sentence_features[0])["sentence_embedding"]
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
            loss = embeddings.sum() * 0.0

        if self.diagnostic_logger is not None:
            with torch.no_grad():
                self.diagnostic_logger.maybe_log(stats, float(loss.detach().item()))

        return loss


def _use_custom_diagnostics(cfg: ContrastiveConfig) -> bool:
    if cfg.triplet_log_diagnostics:
        return True
    impl = (cfg.triplet_implementation or "sentence_transformers").strip().lower()
    return impl == "custom_diagnostics"


def _resolve_soft_margin(cfg: ContrastiveConfig) -> bool:
    loss_type = (cfg.triplet_loss_type or "soft_margin").strip().lower()
    if loss_type in ("hard", "batch_hard"):
        return False
    if loss_type in ("soft_margin", "soft", "batch_hard_soft_margin"):
        return True
    if cfg.batch_triplet_margin is not None:
        return False
    return True


def build_batch_triplet_loss(
    cfg: ContrastiveConfig,
    model: Any,
    *,
    diagnostic_logger: Optional[TripletDiagnosticLogger] = None,
):
    """
    Factory loss Batch Triplet.

  - Par défaut (``log_diagnostics=false``) : ``BatchHardSoftMarginTripletLoss`` ST natif.
  - Sinon : ``BatchTripletLossWithDiagnostics`` (cosine/euclidean + monitoring).
    """
    if not _use_custom_diagnostics(cfg):
        return build_batch_hard_soft_margin_loss(cfg, model)

    soft_margin = _resolve_soft_margin(cfg)
    margin = cfg.batch_triplet_margin if not soft_margin else None
    return BatchTripletLossWithDiagnostics(
        model,
        distance_metric=cfg.distance_metric,
        soft_margin=soft_margin,
        margin=margin,
        diagnostics_eps=cfg.triplet_diagnostics_eps,
        diagnostic_logger=diagnostic_logger,
    )


def build_batch_hard_soft_margin_loss(cfg: ContrastiveConfig, model: Any):
    """Instancie ``BatchHardSoftMarginTripletLoss`` (Sentence Transformers) sans diagnostics."""
    from sentence_transformers import losses

    soft_margin = _resolve_soft_margin(cfg)
    if not soft_margin:
        kwargs: dict[str, Any] = {
            "model": model,
            "distance_metric": resolve_triplet_distance(cfg.distance_metric),
        }
        margin = cfg.batch_triplet_margin
        if margin is None:
            margin = 5.0
        kwargs["triplet_margin"] = float(margin)
        return losses.BatchHardTripletLoss(**kwargs)

    kwargs = {
        "model": model,
        "distance_metric": resolve_triplet_distance(cfg.distance_metric),
    }
    margin = cfg.batch_triplet_margin
    if margin is not None:
        sig = inspect.signature(losses.BatchHardSoftMarginTripletLoss.__init__)
        if "triplet_margin" in sig.parameters:
            kwargs["triplet_margin"] = float(margin)
    return losses.BatchHardSoftMarginTripletLoss(**kwargs)
