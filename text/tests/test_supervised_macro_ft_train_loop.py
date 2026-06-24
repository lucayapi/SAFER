"""Tests boucle d'entraînement supervised_macro_ft."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from supervised_macro_ft.model import SupervisedMacroModel
from supervised_macro_ft.train_loop import fit_model


class _TinyBatchDataset(Dataset):
    def __init__(self, n: int = 24, seq: int = 8) -> None:
        self.n = n
        self.seq = seq

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": torch.randint(0, 50, (self.seq,)),
            "attention_mask": torch.ones(self.seq, dtype=torch.long),
            "label_ids": torch.tensor(idx % 4, dtype=torch.long),
            "index": idx,
        }


def test_fit_model_returns_epoch_history_with_early_stopping():
    model = SupervisedMacroModel(
        backbone_name="__test_dummy__",
        num_classes=4,
        backbone_trainable=False,
        projection="mlp",
        hiddim=16,
        dropout=0.0,
    )
    train_loader = DataLoader(_TinyBatchDataset(), batch_size=8, shuffle=True)
    val_loader = DataLoader(_TinyBatchDataset(), batch_size=8, shuffle=False)
    train_cfg = {
        "epochs": 6,
        "early_stopping_patience": 2,
        "selection_metric": "macro_f1",
        "lr_backbone": 1e-3,
        "lr_head": 1e-2,
    }
    _, best_metrics, history = fit_model(
        model,
        train_loader,
        val_loader,
        train_cfg=train_cfg,
        device=torch.device("cpu"),
        run_label="test_fold",
    )
    assert len(history) >= 1
    assert "epoch" in history[0] and "train_loss" in history[0]
    assert "val_macro_f1" in history[0]
    assert "epoch" in best_metrics
    assert any(row.get("is_best") for row in history)
