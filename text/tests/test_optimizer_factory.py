from argparse import Namespace

import torch

from scgm_text.optimizers import build_optimizer
from scgm_text.schedulers import build_scheduler, step_scheduler
from scgm_text.scgm_embedding_model import SCGMEmbeddingNet


def _args(**kwargs):
    base = dict(optimizer="adamw", lr=1e-3, weight_decay=1e-4, momentum=0.9, scheduler="none", num_cycles=10)
    base.update(kwargs)
    return Namespace(**base)


def test_build_adamw_and_sgd():
    model = SCGMEmbeddingNet(32, 16, 4, 8, projection="linear")
    opt_a = build_optimizer(model, _args(optimizer="adamw", head_lr=1e-3))
    assert isinstance(opt_a, torch.optim.AdamW)
    assert len(opt_a.param_groups) >= 1
    opt_s = build_optimizer(model, _args(optimizer="sgd", head_lr=1e-3))
    assert isinstance(opt_s, torch.optim.SGD)


def test_cosine_scheduler_steps():
    model = SCGMEmbeddingNet(32, 16, 4, 8, projection="linear")
    args = _args(optimizer="sgd", scheduler="cosine", lr=0.1, num_cycles=2)
    opt = build_optimizer(model, args)
    assert build_scheduler(opt, args) == "cosine"
    lr1 = step_scheduler(opt, args, epoch=1, total_epochs=4)
    lr2 = step_scheduler(opt, args, epoch=2, total_epochs=4)
    assert lr1 > 0 and lr2 > 0
