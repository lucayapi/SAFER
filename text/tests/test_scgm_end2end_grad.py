"""pytest: SCGM end2end gradients and optimizer groups."""

from argparse import Namespace

import torch

from scgm_text.optimizers import build_optimizer
from scgm_text.scgm_text_model import SCGMTextModel
from scgm_text.training_diagnostics import assert_scgm_trainability, grad_norm_by_group


def _batch(n: int = 4) -> dict:
    return {
        "input_ids": torch.randint(1, 50, (n, 12)),
        "attention_mask": torch.ones(n, 12, dtype=torch.long),
        "label_ids": torch.randint(0, 4, (n,)),
        "indices": torch.arange(n),
    }


def _cfg() -> Namespace:
    return Namespace(
        optimizer="sgd",
        momentum=0.9,
        lr_backbone=1e-5,
        lr_projector=1e-4,
        lr_head=1e-3,
        weight_decay_backbone=0.01,
        weight_decay_projector=1e-4,
        weight_decay_head=1e-4,
    )


def test_end2end_backbone_receives_gradients():
    model = SCGMTextModel(
        hiddim=32,
        num_classes=4,
        num_subclasses=8,
        backbone_model_name_or_path="__test_dummy__",
        projection="linear",
        backbone_trainable=True,
    )
    batch = _batch()
    q = torch.zeros(4, 8)
    q[torch.arange(4), torch.randint(0, 8, (4,))] = 1.0
    y = torch.zeros(4, 4)
    y[torch.arange(4), batch["label_ids"]] = 1.0

    model.train()
    features = model(batch)
    loss, *_ = model.loss(features, q, y, 0.1, 0.5)
    loss.backward()
    norms = grad_norm_by_group(model)
    assert norms["backbone"] > 0.0
    assert norms["projector"] > 0.0
    assert norms["head"] > 0.0


def test_frozen_backbone_no_grad():
    model = SCGMTextModel(
        hiddim=32,
        num_classes=4,
        num_subclasses=8,
        backbone_model_name_or_path="__test_dummy__",
        projection="linear",
        backbone_trainable=False,
    )
    assert not model.has_trainable_backbone
    opt = build_optimizer(model, _cfg())
    assert_scgm_trainability(model, opt, expect_backbone_trainable=False)

    batch = _batch()
    q = torch.zeros(4, 8)
    q[torch.arange(4), torch.randint(0, 8, (4,))] = 1.0
    y = torch.zeros(4, 4)
    y[torch.arange(4), batch["label_ids"]] = 1.0

    model.train()
    features = model(batch)
    loss, *_ = model.loss(features, q, y, 0.1, 0.5)
    loss.backward()
    norms = grad_norm_by_group(model)
    assert norms["backbone"] == 0.0
    assert norms["projector"] > 0.0
    assert norms["head"] > 0.0


def test_partial_last_two_layers_trainable():
    model = SCGMTextModel(
        hiddim=32,
        num_classes=4,
        num_subclasses=8,
        backbone_model_name_or_path="__test_dummy__",
        projection="linear",
        backbone_trainable=True,
        train_last_n_layers=2,
    )
    flags = [
        any(p.requires_grad for p in layer.parameters())
        for layer in model.backbone.model.layers
    ]
    assert flags == [False, False, True, True]


def test_optimizer_three_param_groups():
    model = SCGMTextModel(
        hiddim=32,
        num_classes=4,
        num_subclasses=8,
        backbone_model_name_or_path="__test_dummy__",
        projection="mlp",
        backbone_trainable=True,
    )
    opt = build_optimizer(model, _cfg())
    names = {g.get("name") for g in opt.param_groups}
    assert names == {"backbone", "projector", "head"}


def test_optimizer_frozen_two_groups():
    model = SCGMTextModel(
        hiddim=32,
        num_classes=4,
        num_subclasses=8,
        backbone_model_name_or_path="__test_dummy__",
        projection="mlp",
        backbone_trainable=False,
    )
    opt = build_optimizer(model, _cfg())
    names = {g.get("name") for g in opt.param_groups}
    assert names == {"projector", "head"}
