"""Utilitaires batch (device, forward features) — end2end dict batches only."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


def batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def forward_features(model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
    if not isinstance(batch, dict):
        raise TypeError(f"SCGM end2end expects dict batch, got {type(batch)!r}")
    return model(batch)


def unpack_batch(batch: Dict[str, Any]) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
    label_ids = batch["label_ids"]
    indices = batch["indices"]
    return batch, label_ids, indices
