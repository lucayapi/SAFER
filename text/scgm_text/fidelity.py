"""SCGM strict-fidelity training (end2end text, AdamW + cosine)."""

from __future__ import annotations

from argparse import Namespace
from typing import Any, Dict

from scgm_text.training_diagnostics import END2END_BANNER, describe_fidelity_mode

_IGNORED_CONFIG_KEYS = frozenset(
    {
        "emb_csv",
        "input_mode",
        "precomputed_embeddings",
        "freeze_backbone",
        "use_self_distillation",
        "teacher_mode",
        "kd_t",
        "with_mlp",
        "momentum",
    }
)


def _set(ns: Namespace, key: str, value: Any) -> None:
    setattr(ns, key, value)


def _set_default(ns: Namespace, key: str, value: Any) -> None:
    if getattr(ns, key, None) is None:
        setattr(ns, key, value)


def apply_scgm_strict_defaults(args: Namespace) -> None:
    """Valeurs par défaut si absentes (ne remplace pas une config YAML explicite)."""
    _set(args, "fidelity_mode", getattr(args, "fidelity_mode", "strict_fidelity"))
    _set_default(args, "optimizer", "adamw")
    _set_default(args, "lr_backbone", 5e-6)
    _set_default(args, "lr_projector", 5e-4)
    _set_default(args, "lr_head", 1e-3)
    _set(args, "scheduler", getattr(args, "scheduler", "cosine"))
    _set(args, "projection", getattr(args, "projection", "mlp"))
    _set(args, "pooling", getattr(args, "pooling", "mean"))
    _set(args, "beta", getattr(args, "beta", 1.0))
    _set(args, "beta1", getattr(args, "beta1", getattr(args, "beta", 1.0)))
    _set(args, "beta2", getattr(args, "beta2", getattr(args, "beta", 1.0)))
    _set(args, "beta3", getattr(args, "beta3", getattr(args, "beta", 1.0)))
    _set(args, "gradient_accumulation_steps", int(getattr(args, "gradient_accumulation_steps", 1)))
    _set(args, "gradient_checkpointing", bool(getattr(args, "gradient_checkpointing", False)))


def flatten_config_yaml(data: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for section in ("data", "model", "training"):
        block = data.get(section)
        if isinstance(block, dict):
            flat.update(block)
    for key, value in data.items():
        if key not in ("model", "training", "data") and not isinstance(value, dict):
            flat[key] = value
    return flat


def apply_config_to_args(args: Namespace, flat: Dict[str, Any]) -> None:
    if "backbone_trainable" in flat or "backbone-trainable" in flat:
        args._config_had_backbone_trainable = True  # type: ignore[attr-defined]
    for key, value in flat.items():
        key_norm = key.replace("-", "_")
        if key_norm == "freeze_backbone" and not getattr(
            args, "_config_had_backbone_trainable", False
        ):
            from scgm_text.config_parsing import str2bool

            args.backbone_trainable = not str2bool(value)
            continue
        if key_norm in _IGNORED_CONFIG_KEYS:
            continue
        if key_norm == "n_folds":
            args.kfold = int(value)
            continue
        if key_norm == "test_corpus" and value:
            args.test_corpus = str(value)
            continue
        if hasattr(args, key_norm):
            setattr(args, key_norm, value)


__all__ = [
    "END2END_BANNER",
    "describe_fidelity_mode",
    "apply_scgm_strict_defaults",
    "flatten_config_yaml",
    "apply_config_to_args",
    "_IGNORED_CONFIG_KEYS",
]
