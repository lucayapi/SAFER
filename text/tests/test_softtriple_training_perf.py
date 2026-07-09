"""Tests optimisations entraînement SoftTriple (val fusionnée, AMP helper)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.hf_training_common import resolve_autocast_dtype
from contrastive_methods.training_softtriple import (
    _dataloader_kwargs,
    _run_val_epoch_with_geometry,
)


class _FakeEncoder(torch.nn.Module):
    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.embedding_dim = dim

    def forward(self, inputs):
        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
        else:
            input_ids = inputs
        b = input_ids.shape[0]
        return torch.ones(b, self.embedding_dim, device=input_ids.device)


class _FakeLoss(torch.nn.Module):
    def forward(self, emb, labels):
        loss = emb.sum() * 0.0 + labels.float().mean() * 0.01
        return loss, {"loss_total": float(loss.detach())}


def test_resolve_autocast_dtype_cpu():
    assert resolve_autocast_dtype("cpu") is None


def test_dataloader_kwargs_cpu_empty():
    assert _dataloader_kwargs("cpu") == {}


def test_dataloader_kwargs_cuda_pin_memory():
    kw = _dataloader_kwargs("cuda:0")
    assert kw.get("pin_memory") is True
    assert kw.get("num_workers", 0) >= 1


def test_run_val_epoch_with_geometry_single_pass():
    cfg = ContrastiveConfig(
        method_name="softtriple",
        dataset_path=Path("."),
        label_col="pred_label",
    )
    val_df = pd.DataFrame(
        {
            "sentence": ["a", "b"],
            "pred_label": ["A0", "A1"],
            "label_id": [0, 1],
        }
    )
    ds = TensorDataset(
        torch.tensor([[1, 2], [3, 4]]),
        torch.tensor([[1, 1], [1, 1]]),
        torch.tensor([0, 1]),
    )

    def collate(batch):
        input_ids = torch.stack([b[0] for b in batch])
        attention_mask = torch.stack([b[1] for b in batch])
        labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    loader = DataLoader(ds, batch_size=2, collate_fn=collate)
    encoder = _FakeEncoder()
    loss_module = _FakeLoss()
    val_loss, geom = _run_val_epoch_with_geometry(
        encoder,  # type: ignore[arg-type]
        loss_module,  # type: ignore[arg-type]
        loader,
        val_df,
        cfg,
        torch.device("cpu"),
    )
    assert np.isfinite(val_loss)
    assert "eta2_macro_balanced_perc" in geom
