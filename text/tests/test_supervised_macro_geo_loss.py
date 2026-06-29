"""Tests loss géométrique supervised_macro_geo_ft."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from supervised_macro_ft.geometry_loss import similarity_preservation_loss
from supervised_macro_ft.model import SupervisedMacroModel
from supervised_macro_ft.train_loop import fit_model


def test_similarity_preservation_loss_shape_and_diag_mask():
    h = torch.randn(8, 32)
    z = torch.randn(8, 16, requires_grad=True)
    loss = similarity_preservation_loss(h, z, remove_diag=True)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    loss.backward()
    assert z.grad is not None


def test_similarity_preservation_loss_batch_size_one_returns_zero():
    h = torch.randn(1, 32)
    z = torch.randn(1, 16)
    loss = similarity_preservation_loss(h, z)
    assert loss.item() == 0.0


def test_similarity_preservation_loss_no_grad_on_frozen_h():
    h = torch.randn(6, 32)  # embeddings Qwen gelés (pas de grad)
    z = torch.randn(6, 16, requires_grad=True)
    loss = similarity_preservation_loss(h, z)
    loss.backward()
    assert z.grad is not None


def test_forward_with_latents_hidden_batch():
    model = SupervisedMacroModel(
        backbone_name="__test_dummy__",
        num_classes=4,
        backbone_trainable=False,
        projection="mlp",
        hiddim=16,
        dropout=0.0,
    )
    batch = {
        "hidden": torch.randn(4, model.backbone.hidden_size),
        "label_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
    }
    logits, z, h = model.forward_with_latents(batch)
    assert logits.shape == (4, 4)
    assert z.shape == (4, 16)
    assert h.shape == (4, model.backbone.hidden_size)


class _HiddenBatchDataset(Dataset):
    def __init__(self, hidden_dim: int, n: int = 16) -> None:
        self.hidden = torch.randn(n, hidden_dim)
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        return {
            "hidden": self.hidden[idx],
            "label_ids": torch.tensor(idx % 4, dtype=torch.long),
        }


def _collate_hidden(items):
    return {
        "hidden": torch.stack([it["hidden"] for it in items]),
        "label_ids": torch.stack([it["label_ids"] for it in items]),
    }


def test_fit_model_with_lambda_geo_records_geo_loss():
    model = SupervisedMacroModel(
        backbone_name="__test_dummy__",
        num_classes=4,
        backbone_trainable=False,
        projection="mlp",
        hiddim=16,
        dropout=0.0,
    )
    hidden_dim = int(model.backbone.hidden_size)
    train_loader = DataLoader(
        _HiddenBatchDataset(hidden_dim),
        batch_size=8,
        shuffle=True,
        collate_fn=_collate_hidden,
    )
    val_loader = DataLoader(
        _HiddenBatchDataset(hidden_dim),
        batch_size=8,
        shuffle=False,
        collate_fn=_collate_hidden,
    )
    _, _, history = fit_model(
        model,
        train_loader,
        val_loader,
        train_cfg={
            "epochs": 2,
            "early_stopping_patience": 5,
            "selection_metric": "macro_f1",
            "lr_projector": 1e-2,
            "lr_head": 1e-2,
            "lambda_geo": 0.1,
        },
        device=torch.device("cpu"),
        run_label="geo_test",
    )
    assert history
    assert "train_loss_ce" in history[0]
    assert "train_loss_geo" in history[0]
    assert history[0]["train_loss_geo"] >= 0.0
