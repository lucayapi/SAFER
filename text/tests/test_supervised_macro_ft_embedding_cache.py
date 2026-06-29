"""Tests cache embeddings backbone supervised_macro_ft."""

from __future__ import annotations

import numpy as np
import torch

from supervised_macro_ft.embedding_cache import (
    BackboneHiddenDataset,
    collate_hidden_batch,
    encode_projected_matrix,
    predict_from_hidden_matrix,
    should_cache_backbone_embeddings,
)
from supervised_macro_ft.model import SupervisedMacroModel


def test_should_cache_backbone_embeddings_only_when_frozen():
    assert should_cache_backbone_embeddings({"backbone_trainable": False, "cache_backbone_embeddings": True})
    assert not should_cache_backbone_embeddings({"backbone_trainable": True})
    assert not should_cache_backbone_embeddings(
        {"backbone_trainable": False, "cache_backbone_embeddings": False}
    )


def test_forward_on_hidden_batch_skips_backbone():
    model = SupervisedMacroModel(
        backbone_name="__test_dummy__",
        num_classes=4,
        backbone_trainable=False,
        projection="linear",
        hiddim=16,
    )
    h = torch.randn(5, model.backbone.hidden_size)
    batch = {"hidden": h, "label_ids": torch.tensor([0, 1, 2, 3, 0])}
    logits = model(batch)
    assert logits.shape == (5, 4)
    loss = torch.nn.functional.cross_entropy(logits, batch["label_ids"])
    loss.backward()
    assert all(p.grad is None for p in model.backbone.parameters())


def test_backbone_hidden_dataset_and_fast_predict():
    model = SupervisedMacroModel(
        backbone_name="__test_dummy__",
        num_classes=4,
        backbone_trainable=False,
        projection="linear",
        hiddim=8,
    )
    hidden = np.random.randn(12, model.backbone.hidden_size).astype(np.float32)
    labels = np.arange(12, dtype=np.int64) % 4
    ds = BackboneHiddenDataset(hidden, labels, indices=np.array([0, 2, 4]))
    batch = collate_hidden_batch([ds[i] for i in range(3)])
    logits = model(batch)
    assert logits.shape == (3, 4)

    device = torch.device("cpu")
    z = encode_projected_matrix(model, hidden, batch_size=4, device=device)
    assert z.shape == (12, 8)
    pred, probs, _, _, _ = predict_from_hidden_matrix(
        model, hidden, macros=["A0", "A1", "B", "C"], batch_size=4, device=device
    )
    assert len(pred) == 12 and probs.shape == (12, 4)
