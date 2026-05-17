"""Pooling des sorties backbone (CLS / mean)."""

from __future__ import annotations

import torch


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def cls_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    del attention_mask
    return last_hidden_state[:, 0, :]


def pool_outputs(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: str = "mean",
) -> torch.Tensor:
    pooling = str(mode).strip().lower()
    if pooling == "mean":
        return mean_pool(last_hidden_state, attention_mask)
    if pooling == "cls":
        return cls_pool(last_hidden_state, attention_mask)
    raise ValueError(f"pooling inconnu : {mode!r} (attendu cls ou mean)")
