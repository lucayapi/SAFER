"""Utilitaires batch (device, forward features)."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn


def batch_to_device(
    batch: Union[torch.Tensor, Dict[str, torch.Tensor], Tuple],
    device: torch.device,
) -> Union[torch.Tensor, Dict[str, torch.Tensor], Tuple]:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        out: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                out[key] = value.to(device)
            else:
                out[key] = value
        return out
    if isinstance(batch, (list, tuple)):
        return type(batch)(batch_to_device(x, device) for x in batch)
    return batch


def forward_features(model: nn.Module, batch: Any) -> torch.Tensor:
    if isinstance(batch, dict):
        return model(batch)
    return model(batch)


def unpack_batch(
    batch: Any,
) -> Tuple[Any, torch.Tensor, torch.Tensor]:
    """Retourne (features_input, label_ids, indices)."""
    if isinstance(batch, dict):
        label_ids = batch["label_ids"]
        indices = batch["indices"]
        return batch, label_ids, indices
    embeddings, label_ids, indices = batch
    return embeddings, label_ids, indices
