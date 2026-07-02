"""Standardisation des embeddings backbone (équivalent sklearn StandardScaler)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch


def should_standardize_backbone(model_cfg: Mapping[str, Any]) -> bool:
    return bool(model_cfg.get("standardize_backbone", False))


@dataclass
class BackboneScaler:
    """Fit sur le train : (h - mean) / std par dimension (comme StandardScaler)."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, hidden: np.ndarray, train_indices: Sequence[int]) -> BackboneScaler:
        h = np.asarray(hidden, dtype=np.float64)
        idx = np.asarray(list(train_indices), dtype=np.int64)
        if h.ndim != 2:
            raise ValueError(f"hidden doit être 2D, reçu {h.shape}")
        if len(idx) == 0:
            raise ValueError("train_indices vide pour BackboneScaler.fit")
        train = h[idx]
        mean = train.mean(axis=0)
        std = train.std(axis=0)
        std = np.where(std > 1e-8, std, 1.0)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform_numpy(self, hidden: np.ndarray) -> np.ndarray:
        h = np.asarray(hidden, dtype=np.float32)
        return (h - self.mean) / self.std

    def transform_tensor(self, hidden: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=hidden.device, dtype=hidden.dtype)
        std = torch.as_tensor(self.std, device=hidden.device, dtype=hidden.dtype)
        return (hidden - mean) / std

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BackboneScaler:
        mean = np.asarray(payload["mean"], dtype=np.float32)
        std = np.asarray(payload["std"], dtype=np.float32)
        if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
            raise ValueError("backbone_scaler invalide : mean/std doivent être 1D de même taille")
        return cls(mean=mean, std=std)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> Optional[BackboneScaler]:
        if not should_standardize_backbone(cfg):
            return None
        payload = cfg.get("backbone_scaler")
        if not payload:
            return None
        return cls.from_dict(payload)
