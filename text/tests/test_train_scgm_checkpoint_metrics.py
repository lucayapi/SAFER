"""Tests sélection checkpoint SCGM (eta², pas F1)."""

from __future__ import annotations

import argparse

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from scgm_text.scgm_text_model import SCGMTextModel
from scripts.train_scgm_text import checkpoint_selection_score, evaluate_split


class _TinyTextDS(Dataset):
    def __len__(self):
        return 80

    def __getitem__(self, i):
        return {
            "text": "sample",
            "label": int(i % 4),
            "group": "g",
            "row_id": i,
            "index": i,
        }


def _collate(batch):
    n = len(batch)
    return {
        "input_ids": torch.randint(1, 50, (n, 8)),
        "attention_mask": torch.ones(n, 8, dtype=torch.long),
        "label_ids": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "indices": torch.tensor([b["index"] for b in batch], dtype=torch.long),
    }


def _model():
    return SCGMTextModel(
        hiddim=32,
        num_classes=4,
        num_subclasses=8,
        backbone_model_name_or_path="__test_dummy__",
        projection="linear",
    )


def _loader() -> DataLoader:
    return DataLoader(_TinyTextDS(), batch_size=16, collate_fn=_collate)


def test_evaluate_split_returns_eta2_keys():
    model = _model()
    metrics, _, _, _, _ = evaluate_split(
        model, _loader(), torch.device("cpu"), tau=0.1, n_class=4, prefix="val"
    )
    assert "val_eta2_macro_balanced" in metrics
    assert "val_eta2_weighted" in metrics
    assert np.isfinite(metrics["val_eta2_macro_balanced"])
    assert "val_macro_f1" not in metrics
    assert "rankme_global" not in metrics


def test_evaluate_split_classifier_diagnostics_optional():
    model = _model()
    metrics, _, _, _, _ = evaluate_split(
        model,
        _loader(),
        torch.device("cpu"),
        tau=0.1,
        n_class=4,
        prefix="val",
        compute_classifier_diagnostics=True,
    )
    assert "val_macro_f1" in metrics
    assert np.isfinite(metrics["val_macro_f1"])


def test_checkpoint_selection_prefers_higher_eta2():
    low = {"val_eta2_macro_balanced": 0.1}
    high = {"val_eta2_macro_balanced": 0.4}
    assert checkpoint_selection_score(high, "eta2_macro_balanced") > checkpoint_selection_score(
        low, "eta2_macro_balanced"
    )
