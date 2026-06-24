"""Tests modèle supervised_macro_ft."""

from __future__ import annotations

import torch

from supervised_macro_ft.model import SupervisedMacroModel


def test_supervised_macro_model_forward_and_probs_with_projector():
    model = SupervisedMacroModel(
        backbone_name="__test_dummy__",
        num_classes=4,
        backbone_trainable=False,
        projection="mlp",
        hiddim=32,
        dropout=0.0,
    )
    batch = 6
    seq = 8
    input_ids = torch.randint(0, 50, (batch, seq))
    attention_mask = torch.ones(batch, seq, dtype=torch.long)
    logits = model.forward_logits(input_ids, attention_mask)
    assert logits.shape == (batch, 4)
    z = model.encode(input_ids, attention_mask)
    assert z.shape == (batch, 32)
    probs, _ = model.predict_proba(input_ids, attention_mask)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch), atol=1e-5)

    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3, 0, 1]))
    loss.backward()
    assert any(p.grad is not None for p in model.projector.parameters())
    assert any(p.grad is not None for p in model.classifier.parameters())
    assert all(p.grad is None for p in model.backbone.parameters())


def test_supervised_macro_model_legacy_without_projector():
    model = SupervisedMacroModel(
        backbone_name="__test_dummy__",
        num_classes=4,
        backbone_trainable=True,
        projection=None,
    )
    batch = 4
    seq = 6
    input_ids = torch.randint(0, 50, (batch, seq))
    attention_mask = torch.ones(batch, seq, dtype=torch.long)
    z = model.encode(input_ids, attention_mask)
    assert z.shape == (batch, model.backbone.hidden_size)
    assert not model.use_projector
