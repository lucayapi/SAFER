"""Validation de configuration supervised_macro_ft au démarrage."""

from __future__ import annotations

import logging
from typing import Any, Mapping

import torch

from supervised_macro_ft.backbone_scaler import should_standardize_backbone
from supervised_macro_ft.class_balance import resolve_train_balance
from supervised_macro_ft.model import SupervisedMacroModel, model_kwargs_from_cfg, validate_macro_ft_projection
from supervised_macro_ft.run_logging import (
    log_effective_config,
    log_trainable_param_counts,
    resolve_effective_use_amp,
)

logger = logging.getLogger(__name__)


def _count_trainable_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def validate_macro_ft_startup(
    model_cfg: Mapping[str, Any],
    train_cfg: Mapping[str, Any],
    *,
    device: torch.device,
    n_samples: int | None = None,
    backbone_hidden_available: bool = False,
) -> tuple[bool, str | None]:
    """Valide la config FT et log le récapitulatif."""
    validate_macro_ft_projection(model_cfg.get("projection", "mlp_sklearn"))
    use_oversampling, class_weight_mode = resolve_train_balance(model_cfg)

    backbone_trainable = bool(model_cfg.get("backbone_trainable", True))
    if backbone_trainable:
        if bool(model_cfg.get("cache_backbone_embeddings", True)):
            logger.warning(
                "[macro_ft] backbone_trainable=true : cache_backbone_embeddings ignoré "
                "(Qwen recalculé à chaque epoch)."
            )
        if should_standardize_backbone(model_cfg):
            raise ValueError(
                "standardize_backbone=true est incompatible avec backbone_trainable=true."
            )

    log_effective_config(
        model_cfg,
        train_cfg,
        device=device,
        use_oversampling=use_oversampling,
        class_weight_mode=class_weight_mode,
        n_samples=n_samples,
        backbone_hidden_available=backbone_hidden_available,
    )

    try:
        probe = SupervisedMacroModel(**model_kwargs_from_cfg(model_cfg)).to(device)
        log_trainable_param_counts(
            n_backbone=_count_trainable_params(probe.backbone),
            n_projector=_count_trainable_params(probe.projector),
            n_head=_count_trainable_params(probe.classifier),
        )
        del probe
    except Exception as exc:
        logger.warning("[macro_ft] Impossible de compter les paramètres trainables : %s", exc)

    use_amp, amp_reason = resolve_effective_use_amp(
        train_cfg,
        backbone_trainable=backbone_trainable,
        device=device,
    )
    logger.info("[macro_ft] AMP effective : use_amp=%s (%s)", use_amp, amp_reason)

    return use_oversampling, class_weight_mode


def resolve_use_amp(
    train_cfg: Mapping[str, Any],
    *,
    backbone_trainable: bool,
    device: torch.device,
) -> bool:
    enabled, _ = resolve_effective_use_amp(
        train_cfg,
        backbone_trainable=backbone_trainable,
        device=device,
    )
    return enabled
