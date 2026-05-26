"""Parsing robuste des options CLI/YAML SCGM."""

from __future__ import annotations

import argparse
from typing import Any


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("yes", "true", "t", "1"):
        return True
    if text in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def normalize_backbone_trainability(
    backbone_trainable: bool,
    train_last_n_layers: int | None,
) -> tuple[bool, int | None]:
    """Résout conflits backbone_trainable / train_last_n_layers."""
    if train_last_n_layers == 0:
        if backbone_trainable:
            print(
                "[SCGM] train_last_n_layers=0 -> backbone_trainable=False",
                flush=True,
            )
        backbone_trainable = False
        train_last_n_layers = None
    elif not backbone_trainable:
        if train_last_n_layers is not None and train_last_n_layers > 0:
            print(
                "[SCGM WARN] backbone_trainable=false ignores "
                f"train_last_n_layers={train_last_n_layers}",
                flush=True,
            )
        train_last_n_layers = None
    return backbone_trainable, train_last_n_layers
