"""Tests projecteurs macro FT (linear, mlp_sklearn)."""

from __future__ import annotations

import pytest
import torch

from supervised_macro_ft.model import (
    SupervisedMacroModel,
    model_kwargs_from_cfg,
    validate_macro_ft_projection,
)


def _dummy_batch(batch: int = 4, seq: int = 8):
    return (
        torch.randint(0, 50, (batch, seq)),
        torch.ones(batch, seq, dtype=torch.long),
    )


@pytest.mark.parametrize(
    "projection,hiddim,extra",
    [
        ("linear", 32, {}),
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


def test_validate_macro_ft_projection_rejects_ln_gelu():
    with pytest.raises(ValueError, match="projection FT non supportée"):
        validate_macro_ft_projection("ln_gelu")


def test_validate_macro_ft_projection_null_disables_projector():
    assert validate_macro_ft_projection(None) is None
    assert validate_macro_ft_projection("null") is None
    assert validate_macro_ft_projection("none") is None


def test_model_kwargs_from_cfg_null_projection():
    kw = model_kwargs_from_cfg(
        {
            "backbone_name": "x",
            "projection": "null",
        }
    )
    assert kw["projection"] is None


def test_model_kwargs_from_cfg_mlp_sklearn_forces_hiddim_128():
    kw = model_kwargs_from_cfg(
        {
            "backbone_name": "x",
            "projection": "mlp_sklearn",
            "hiddim": 512,
        }
    )
    assert kw["projection"] == "mlp_sklearn"
    assert kw["hiddim"] == 128


def test_backbone_trainable_has_grad():
    model = SupervisedMacroModel(
        backbone_name="__test_dummy__",
        num_classes=4,
        backbone_trainable=True,
        projection="linear",
        hiddim=32,
    )
    input_ids, attention_mask = _dummy_batch(batch=2)
    h = model._encode_backbone(input_ids, attention_mask)
    assert h.requires_grad
