from argparse import Namespace

import pytest
import torch

from scgm_text.batch_utils import forward_features
from scgm_text.optimizers import build_optimizer
from scgm_text.scgm_text_model import SCGMTextModel
from scgm_text.training_diagnostics import (
    measure_backbone_weight_change,
    snapshot_backbone_weights,
)


def _cfg(**kwargs):
    base = dict(
        optimizer="adamw",
        lr_backbone=2e-5,
        lr_projector=1e-4,
        lr_head=1e-3,
        weight_decay_backbone=0.01,
        weight_decay_projector=1e-4,
        weight_decay_head=1e-4,
        n_class=4,
        n_subclass=8,
        tau=0.1,
        alpha=0.5,
    )
    base.update(kwargs)
    return Namespace(**base)


def _text_batch(batch_size: int = 4) -> dict:
    return {
        "input_ids": torch.randint(1, 50, (batch_size, 12)),
        "attention_mask": torch.ones(batch_size, 12, dtype=torch.long),
        "label_ids": torch.randint(0, 4, (batch_size,)),
        "indices": torch.arange(batch_size),
    }


def _model(projection: str = "linear") -> SCGMTextModel:
    return SCGMTextModel(
        hiddim=32,
        num_classes=4,
        num_subclasses=8,
        projection=projection,
        backbone_model_name_or_path="__test_dummy__",
    )


def test_backbone_has_trainable_params():
    model = _model()
    trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    assert trainable > 0
    assert model.has_trainable_backbone


def test_backbone_updates_after_step():
    model = _model()
    cfg = _cfg()
    optimizer = build_optimizer(model, cfg)
    batch = _text_batch()
    q = torch.zeros(4, 8)
    q[torch.arange(4), torch.randint(0, 8, (4,))] = 1.0
    y = torch.zeros(4, 4)
    y[torch.arange(4), batch["label_ids"]] = 1.0

    before = snapshot_backbone_weights(model)
    features = forward_features(model, batch)
    loss, *_ = model.loss(features, q, y, cfg.tau, cfg.alpha)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    change = measure_backbone_weight_change(model, before)
    assert change > 0.0


def test_forward_features_dict_batch():
    model = _model()
    features = forward_features(model, _text_batch())
    assert features.shape == (4, 32)


def test_mlp_projection_output_dim():
    model = _model(projection="mlp")
    features = model(_text_batch())
    assert features.shape[1] == 32
