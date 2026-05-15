"""SCGM-G fidelity vs text-pragmatic training presets."""

from __future__ import annotations

from argparse import Namespace
from typing import Any, Dict


STRICT_FIDELITY_BANNER = """\
Running SCGM-Text in STRICT FIDELITY MODE.
This mode follows official SCGM-G training choices when applicable:
optimizer=SGD, momentum=0.9, scheduler=cosine, E-step Sinkhorn.
Note: input remains fixed text embeddings, not images, so this is not a full image pipeline reproduction.
"""

PRAGMATIC_BANNER = """\
Running SCGM-Text in TEXT PRAGMATIC MODE.
This mode keeps the SCGM-G objective but uses optimizer/training choices adapted to fixed text embeddings.
"""


def _set(ns: Namespace, key: str, value: Any) -> None:
    setattr(ns, key, value)


def apply_scgm_strict_defaults(args: Namespace) -> None:
    _set(args, "fidelity_mode", "strict")
    _set(args, "optimizer", "sgd")
    _set(args, "momentum", 0.9)
    _set(args, "weight_decay", 1e-4)
    _set(args, "scheduler", "cosine")
    _set(args, "num_cycles", getattr(args, "num_cycles", 10))
    _set(args, "lr", getattr(args, "lr", 0.03))
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
    _set(args, "weight_decay", getattr(args, "weight_decay", 1e-4))
    _set(args, "scheduler", getattr(args, "scheduler", "none"))
    if getattr(args, "projection", None) in (None, ""):
        _set(args, "projection", "linear")
    _set(args, "beta", getattr(args, "beta", 1.0))
    _set(args, "beta1", getattr(args, "beta1", 1.0))
    _set(args, "beta2", getattr(args, "beta2", 1.0))
    _set(args, "beta3", getattr(args, "beta3", 1.0))


def describe_fidelity_mode(args: Namespace) -> str:
    mode = getattr(args, "fidelity_mode", "custom")
    if mode == "strict":
        return STRICT_FIDELITY_BANNER
    if mode == "pragmatic":
        return PRAGMATIC_BANNER
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
