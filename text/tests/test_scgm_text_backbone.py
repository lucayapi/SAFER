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
        lr=1e-3,
        head_lr=1e-3,
        backbone_lr=2e-5,
        weight_decay=1e-4,
        head_weight_decay=1e-4,
        backbone_weight_decay=0.01,
        input_mode="text",
        projection="identity",
        freeze_backbone=False,
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


def _model_text(identity: bool = True, freeze: bool = False) -> SCGMTextModel:
    return SCGMTextModel(
        input_dim=32,
        hiddim=32,
        num_classes=4,
        num_subclasses=8,
        projection="identity" if identity else "linear",
        input_mode="text",
        backbone_model_name_or_path="__test_dummy__",
        freeze_backbone=freeze,
    )


def test_identity_text_backbone_has_trainable_params():
    model = _model_text(identity=True, freeze=False)
    trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    assert trainable > 0
    assert model.has_trainable_backbone


def test_identity_text_backbone_updates_after_step():
    model = _model_text(identity=True, freeze=False)
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


def test_precomputed_identity_has_no_backbone():
    model = SCGMTextModel(
        input_dim=32,
        hiddim=32,
        num_classes=4,
        num_subclasses=8,
        projection="identity",
        input_mode="precomputed_embeddings",
    )
    assert model.backbone is None
    assert not model.has_trainable_backbone


def test_freeze_backbone_true_no_update():
    model = _model_text(identity=True, freeze=True)
    cfg = _cfg(freeze_backbone=True)
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
    assert change == 0.0


def test_optimizer_param_groups():
    model = _model_text(identity=False, freeze=False)
    cfg = _cfg(projection="linear")
    opt = build_optimizer(model, cfg)
    names = {g.get("name") for g in opt.param_groups}
    assert "backbone" in names
    assert "projection" in names
    assert "scgm" in names

    model_pre = SCGMTextModel(
        input_dim=32,
        hiddim=16,
        num_classes=4,
        num_subclasses=8,
        projection="identity",
        input_mode="precomputed_embeddings",
    )
    opt_pre = build_optimizer(model_pre, _cfg(input_mode="precomputed_embeddings", projection="identity"))
    names_pre = {g.get("name") for g in opt_pre.param_groups}
    assert "backbone" not in names_pre
    assert "scgm" in names_pre
