"""SCGM-G fidelity vs text-pragmatic training presets."""

from __future__ import annotations

from argparse import Namespace
from typing import Any, Dict


STRICT_FIDELITY_BANNER = """\
Running SCGM-Text in STRICT FIDELITY MODE (precomputed embeddings).
This mode follows official SCGM-G training choices when applicable:
optimizer=SGD, momentum=0.9, scheduler=cosine, E-step Sinkhorn, projection=mlp.
Note: input_mode=precomputed_embeddings — backbone theta is not in the training graph.
"""

PRAGMATIC_BANNER = """\
Running SCGM-Text in TEXT PRAGMATIC MODE.
AdamW, projection=linear; backbone fine-tuning configurable via freeze_backbone.
"""

STRICT_FINETUNE_IDENTITY_BANNER = """\
Running SCGM-Text in STRICT FINETUNE IDENTITY MODE.
input_mode=text, projection=identity, freeze_backbone=false.
Fine-tune the text backbone directly (h = f_theta(x)), without additional projection.
This is NOT precomputed_identity (fixed embeddings).
"""

PRECOMPUTED_IDENTITY_BANNER = """\
Running SCGM-Text in PRECOMPUTED IDENTITY MODE.
input_mode=precomputed_embeddings, projection=identity.
Fixed precomputed embeddings e; no backbone theta is updated.
This is NOT strict_finetune_identity (trainable backbone).
"""


def _set(ns: Namespace, key: str, value: Any) -> None:
    setattr(ns, key, value)


def apply_scgm_strict_defaults(args: Namespace) -> None:
    _set(args, "fidelity_mode", "strict")
    _set(args, "input_mode", "precomputed_embeddings")
    _set(args, "optimizer", "sgd")
    _set(args, "momentum", 0.9)
    _set(args, "weight_decay", 1e-4)
    _set(args, "scheduler", "cosine")
    _set(args, "num_cycles", getattr(args, "num_cycles", 10))
    _set(args, "lr", getattr(args, "lr", 0.03))
    _set(args, "head_lr", getattr(args, "lr", 0.03))
    if getattr(args, "projection", None) in (None, "", "identity"):
        _set(args, "projection", "mlp")
    _set(args, "beta", getattr(args, "beta", 1.0))
    _set(args, "beta1", getattr(args, "beta1", getattr(args, "beta", 1.0)))
    _set(args, "beta2", getattr(args, "beta2", getattr(args, "beta", 1.0)))
    _set(args, "beta3", getattr(args, "beta3", getattr(args, "beta", 1.0)))
    _set(args, "n_iter_estep", getattr(args, "n_iter_estep", 5))
    _set(args, "kd_t", getattr(args, "kd_t", 4.0))


def apply_text_pragmatic_defaults(args: Namespace) -> None:
    _set(args, "fidelity_mode", "pragmatic")
    _set(args, "optimizer", "adamw")
    _set(args, "lr", getattr(args, "lr", 1e-3))
    _set(args, "head_lr", getattr(args, "head_lr", getattr(args, "lr", 1e-3)))
    _set(args, "backbone_lr", getattr(args, "backbone_lr", 1e-5))
    _set(args, "weight_decay", getattr(args, "weight_decay", 1e-4))
    _set(args, "scheduler", getattr(args, "scheduler", "none"))
    if getattr(args, "projection", None) in (None, ""):
        _set(args, "projection", "linear")
    _set(args, "beta", getattr(args, "beta", 1.0))
    _set(args, "beta1", getattr(args, "beta1", 1.0))
    _set(args, "beta2", getattr(args, "beta2", 1.0))
    _set(args, "beta3", getattr(args, "beta3", 1.0))


def apply_strict_finetune_identity_defaults(args: Namespace) -> None:
    _set(args, "fidelity_mode", "strict_finetune_identity")
    _set(args, "input_mode", "text")
    _set(args, "projection", "identity")
    _set(args, "freeze_backbone", False)
    _set(args, "optimizer", "adamw")
    _set(args, "scheduler", getattr(args, "scheduler", "none"))
    _set(args, "pooling", getattr(args, "pooling", "mean"))
    _set(args, "backbone_lr", getattr(args, "backbone_lr", 2e-5))
    _set(args, "head_lr", getattr(args, "head_lr", 1e-3))
    _set(args, "lr", getattr(args, "head_lr", 1e-3))
    _set(args, "weight_decay", getattr(args, "weight_decay", 1e-4))
    _set(args, "backbone_weight_decay", getattr(args, "backbone_weight_decay", 0.01))
    if getattr(args, "batch_size", 512) >= 64:
        _set(args, "batch_size", 16)


def apply_precomputed_identity_defaults(args: Namespace) -> None:
    _set(args, "fidelity_mode", "precomputed_identity")
    _set(args, "input_mode", "precomputed_embeddings")
    _set(args, "projection", "identity")
    _set(args, "optimizer", getattr(args, "optimizer", "adamw"))
    _set(args, "head_lr", getattr(args, "head_lr", getattr(args, "lr", 1e-3)))
    _set(args, "lr", getattr(args, "head_lr", 1e-3))


def describe_fidelity_mode(args: Namespace) -> str:
    mode = getattr(args, "fidelity_mode", "custom")
    if mode == "strict":
        return STRICT_FIDELITY_BANNER
    if mode == "pragmatic":
        return PRAGMATIC_BANNER
    if mode == "strict_finetune_identity":
        return STRICT_FINETUNE_IDENTITY_BANNER
    if mode == "precomputed_identity":
        return PRECOMPUTED_IDENTITY_BANNER
    return f"Running SCGM-Text in custom mode (fidelity_mode={mode})."


def flatten_config_yaml(data: Dict[str, Any]) -> Dict[str, Any]:
    """Merge model / training / data sections into a flat dict."""
    flat: Dict[str, Any] = {}
    for section in ("data", "model", "training"):
        block = data.get(section)
        if isinstance(block, dict):
            flat.update(block)
    for key, value in data.items():
        if key not in ("model", "training", "data") and not isinstance(value, dict):
            flat[key] = value
    return flat
