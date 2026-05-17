"""Tests sélection checkpoint SCGM (eta², pas F1)."""

from __future__ import annotations

import argparse

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scgm_text.scgm_text_model import SCGMTextModel
from scripts.train_scgm_text import checkpoint_selection_score, evaluate_split


def _args():
    return argparse.Namespace(
        input_mode="precomputed_embeddings",
        projection="identity",
        hiddim=8,
        n_class=4,
        n_subclass=8,
        freeze_backbone=True,
        backbone_model_name_or_path="",
        pooling="mean",
        max_seq_length=32,
    )


def _collate_precomputed(batch):
    emb, y, idx = zip(*batch)
    return {
        "embeddings": torch.stack(emb),
        "label_ids": torch.stack(y),
        "indices": torch.stack(idx),
    }


def _make_loader(n: int = 80, dim: int = 8, seed: int = 0) -> DataLoader:
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.normal(size=(n, dim)), dtype=torch.float32)
    y = torch.tensor(rng.integers(0, 4, size=n), dtype=torch.long)
    idx = torch.arange(n, dtype=torch.long)
    return DataLoader(
        TensorDataset(x, y, idx),
        batch_size=16,
        collate_fn=_collate_precomputed,
    )


def test_evaluate_split_returns_eta2_keys():
    model = SCGMTextModel.from_args(_args(), input_dim=8)
    loader = _make_loader()
    metrics, _, _, _ = evaluate_split(
        model, loader, torch.device("cpu"), tau=0.1, n_class=4, prefix="val"
    )
    assert "val_eta2_macro_balanced" in metrics
    assert "val_eta2_weighted" in metrics
    assert np.isfinite(metrics["val_eta2_macro_balanced"])
    assert "val_macro_f1" not in metrics
    assert "rankme_global" in metrics


def test_evaluate_split_classifier_diagnostics_optional():
    model = SCGMTextModel.from_args(_args(), input_dim=8)
    loader = _make_loader()
    metrics, _, _, _ = evaluate_split(
        model,
        loader,
        torch.device("cpu"),
        tau=0.1,
        n_class=4,
        prefix="val",
        compute_classifier_diagnostics=True,
    )
    assert "val_macro_f1" in metrics
    assert np.isfinite(metrics["val_macro_f1"])


def test_checkpoint_selection_prefers_higher_eta2():
    low = {"val_eta2_macro_balanced": 0.1, "c1_global": 0.05}
    high = {"val_eta2_macro_balanced": 0.4, "c1_global": 0.05}
    assert checkpoint_selection_score(high, "eta2_macro_balanced", 0.01) > checkpoint_selection_score(
        low, "eta2_macro_balanced", 0.01
    )


def test_checkpoint_selection_composite_penalizes_c1():
    a = {"val_eta2_macro_balanced": 0.3, "c1_global": 0.9}
    b = {"val_eta2_macro_balanced": 0.3, "c1_global": 0.1}
    assert checkpoint_selection_score(b, "composite", 0.1) > checkpoint_selection_score(
        a, "composite", 0.1
    )
