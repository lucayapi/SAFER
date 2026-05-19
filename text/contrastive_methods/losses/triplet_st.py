"""
Batch Hard Triplet (Sentence Transformers).

Utilise ``BatchHardSoftMarginTripletLoss`` :
https://sbert.net/docs/sentence_transformer/loss_overview.html

Requiert ``BatchSamplers.GROUP_BY_LABEL`` (voir ``st_common.build_training_arguments``).
"""

from __future__ import annotations

import inspect
from typing import Any

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.st_common import resolve_triplet_distance


def build_batch_hard_soft_margin_loss(cfg: ContrastiveConfig, model: Any):
    """Instancie ``BatchHardSoftMarginTripletLoss`` avec distance et marge optionnelle."""
    from sentence_transformers import losses

    kwargs: dict[str, Any] = {
        "model": model,
        "distance_metric": resolve_triplet_distance(cfg.distance_metric),
    }
    margin = getattr(cfg, "batch_triplet_margin", None)
    if margin is not None:
        sig = inspect.signature(losses.BatchHardSoftMarginTripletLoss.__init__)
        if "triplet_margin" in sig.parameters:
            kwargs["triplet_margin"] = float(margin)
    return losses.BatchHardSoftMarginTripletLoss(**kwargs)
