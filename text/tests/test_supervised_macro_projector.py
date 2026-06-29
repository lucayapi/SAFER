"""Tests projecteurs macro FT (linear, ln_gelu, residual)."""

from __future__ import annotations

import pytest
import torch

from scgm_text.projection import ResidualProjector, build_embedding_projector
from supervised_macro_ft.model import SupervisedMacroModel


def _dummy_batch(batch: int = 4, seq: int = 8):
    return (
        torch.randint(0, 50, (batch, seq)),
        torch.ones(batch, seq, dtype=torch.long),
    )


def test_build_embedding_projector_ln_gelu_shape():
    proj = build_embedding_projector(
        "ln_gelu",
        input_dim=64,
        hiddim=32,
        dropout=0.1,
        proj_hidden=48,
    )
    h = torch.randn(5, 64)
    z = proj(h)
    assert z.shape == (5, 32)


def test_residual_projector_shape():
    proj = ResidualProjector(64, 32, bottleneck=16, alpha=0.1, dropout=0.0)
    h = torch.randn(5, 64)
    z = proj(h)
    assert z.shape == (5, 32)


@pytest.mark.parametrize(
    "projection,hiddim,extra",
    [
        ("linear", 32, {}),
        ("ln_gelu", 32, {"proj_hidden": 48}),
        ("residual", 32, {"proj_bottleneck": 16, "proj_alpha": 0.2}),
        ("mlp_sklearn", 128, {"proj_hidden": 256}),
    ],
)
def test_supervised_macro_model_projections_forward(projection, hiddim, extra):
    model = SupervisedMacroModel(
        backbone_name="__test_dummy__",
        num_classes=4,
        backbone_trainable=False,
        projection=projection,
        hiddim=hiddim,
        dropout=0.0,
        **extra,
    )
    input_ids, attention_mask = _dummy_batch()
    logits = model.forward_logits(input_ids, attention_mask)
    z = model.encode(input_ids, attention_mask)
    assert logits.shape == (4, 4)
    assert z.shape == (4, hiddim)

    hidden = torch.randn(4, model.backbone.hidden_size)
    z_h = model.encode_from_hidden(hidden)
    assert z_h.shape == (4, hiddim)


def test_supervised_macro_model_kwargs_from_cfg_ln_gelu():
    from supervised_macro_ft.model import model_kwargs_from_cfg

    kw = model_kwargs_from_cfg(
        {
            "backbone_name": "x",
            "projection": "residual",
            "hiddim": 256,
            "proj_bottleneck": 128,
            "proj_alpha": 0.1,
        }
    )
    assert kw["projection"] == "residual"
    assert kw["hiddim"] == 256
    assert kw["proj_bottleneck"] == 128


def test_model_kwargs_from_cfg_mlp_sklearn_forces_hiddim_128():
    from supervised_macro_ft.model import model_kwargs_from_cfg

    kw = model_kwargs_from_cfg(
        {
            "backbone_name": "x",
            "projection": "mlp_sklearn",
            "hiddim": 512,
        }
    )
    assert kw["projection"] == "mlp_sklearn"
    assert kw["hiddim"] == 128
