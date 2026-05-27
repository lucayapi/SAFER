import pytest
import torch
from argparse import Namespace

from scgm_text.optimizers import build_optimizer
from scgm_text.scgm_text_model import SCGMTextModel


def test_unknown_optimizer_rejected():
    model = SCGMTextModel(
        hiddim=16,
        num_classes=4,
        num_subclasses=8,
        backbone_model_name_or_path="__test_dummy__",
        projection="linear",
    )
    cfg = Namespace(optimizer="rmsprop")
    with pytest.raises(ValueError, match="only supports optimizer=adamw"):
        build_optimizer(model, cfg)


def test_sgd_rejected():
    model = SCGMTextModel(
        hiddim=16,
        num_classes=4,
        num_subclasses=8,
        backbone_model_name_or_path="__test_dummy__",
        projection="linear",
    )
    cfg = Namespace(optimizer="sgd")
    with pytest.raises(ValueError, match="sgd n'est plus support"):
        build_optimizer(model, cfg)


def test_build_adamw_optimizer_three_groups():
    model = SCGMTextModel(
        hiddim=16,
        num_classes=4,
        num_subclasses=8,
        backbone_model_name_or_path="__test_dummy__",
        projection="linear",
    )
    cfg = Namespace(
        optimizer="adamw",
        lr_backbone=5e-6,
        lr_projector=5e-4,
        lr_head=1e-3,
        weight_decay_backbone=0.01,
        weight_decay_projector=1e-4,
        weight_decay_head=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
    )
    opt = build_optimizer(model, cfg)
    assert isinstance(opt, torch.optim.AdamW)
    assert len(opt.param_groups) == 3
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs == sorted([5e-6, 5e-4, 1e-3])
