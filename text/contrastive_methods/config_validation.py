"""Validation de la config contrastive (backbone / projecteur / post_eval)."""

from __future__ import annotations

from typing import Any

from scgm_text.config_parsing import normalize_backbone_trainability

from contrastive_methods.config import ContrastiveConfig


def validate_contrastive_config(cfg: ContrastiveConfig) -> ContrastiveConfig:
    """Applique quelques règles de cohérence métier sur la config.

    - normalise backbone_trainable / train_last_n_layers
    - si backbone gelé : cache_backbone_embeddings peut rester à True (sinon ignoré)
    - post_eval : interdit oversampling + class_weight=balanced
    """
    bt, last_n = normalize_backbone_trainability(
        bool(cfg.backbone_trainable),
        cfg.train_last_n_layers,
    )
    cfg.backbone_trainable = bt
    cfg.train_last_n_layers = last_n

    if cfg.backbone_trainable:
        cfg.cache_backbone_embeddings = False

    cw_raw: Any = cfg.post_eval_class_weight
    cw = None if cw_raw is None else str(cw_raw).strip().lower()
    if cw in ("none", "null", ""):
        cw = None
    cfg.post_eval_class_weight = cw
    if cfg.post_eval_oversampling and cw == "balanced":
        raise ValueError(
            "Configuration incohérente : post_eval.oversampling=true et "
            "post_eval.class_weight='balanced'. Choisir soit l'oversampling, "
            "soit class_weight=balanced, mais pas les deux."
        )
    return cfg

