import pytest
from argparse import Namespace

from scgm_text.optimizers import build_optimizer
from scgm_text.scgm_text_model import SCGMTextModel


def test_adamw_rejected():
    model = SCGMTextModel(
        hiddim=16,
        num_classes=4,
        num_subclasses=8,
        backbone_model_name_or_path="__test_dummy__",
        projection="linear",
    )
    cfg = Namespace(optimizer="adamw")
    with pytest.raises(ValueError, match="Adam"):
        build_optimizer(model, cfg)


def test_build_optimizer_three_groups():
    model = SCGMTextModel(
        hiddim=16,
        num_classes=4,
        num_subclasses=8,
        backbone_model_name_or_path="__test_dummy__",
        projection="linear",
    )
    cfg = Namespace(
        optimizer="sgd",
        momentum=0.9,
        lr_backbone=1e-5,
        lr_projector=1e-4,
        lr_head=1e-3,
        weight_decay_backbone=0.01,
        weight_decay_projector=1e-4,
        weight_decay_head=1e-4,
    )
    opt = build_optimizer(model, cfg)
    assert len(opt.param_groups) == 3
