"""Tests standardisation backbone (StandardScaler-like)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from supervised_macro_ft.backbone_scaler import BackboneScaler, should_standardize_backbone
from supervised_macro_ft.checkpoint_io import load_checkpoint, save_checkpoint
from supervised_macro_ft.model import SupervisedMacroModel, model_kwargs_from_cfg


def test_should_standardize_backbone_from_cfg():
    assert not should_standardize_backbone({})
    assert should_standardize_backbone({"standardize_backbone": True})
    assert not should_standardize_backbone({"standardize_backbone": False})


def test_backbone_scaler_fit_transform_zero_mean_unit_var():
    rng = np.random.RandomState(0)
    hidden = rng.randn(40, 8).astype(np.float32) * 3.0 + 2.0
    scaler = BackboneScaler.fit(hidden, np.arange(30))
    out = scaler.transform_numpy(hidden[:5])
    train_scaled = scaler.transform_numpy(hidden[:30])
    assert np.allclose(train_scaled.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(train_scaled.std(axis=0), 1.0, atol=1e-5)
    assert out.shape == (5, 8)


def test_model_applies_scaler_only_before_projector():
    model = SupervisedMacroModel(
        backbone_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        num_classes=4,
        backbone_trainable=False,
        projection="linear",
        hiddim=16,
    )
    h = torch.randn(3, model.backbone.hidden_size)
    z_no_scaler = model.encode_from_hidden(h)
    mean = h.mean(dim=0).numpy()
    std = h.std(dim=0).numpy()
    std = np.where(std > 1e-8, std, 1.0)
    model.set_backbone_scaler(BackboneScaler(mean=mean.astype(np.float32), std=std.astype(np.float32)))
    z_scaled = model.encode_from_hidden(h)
    logits, z, h_out = model.forward_with_latents({"hidden": h})
    assert torch.allclose(h_out, h)
    assert not torch.allclose(z_no_scaler, z_scaled)


def test_checkpoint_roundtrip_backbone_scaler(tmp_path):
    model = SupervisedMacroModel(
        backbone_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        num_classes=4,
        backbone_trainable=False,
        projection="linear",
        hiddim=16,
        standardize_backbone=True,
    )
    dim = model.backbone.hidden_size
    model.set_backbone_scaler(
        BackboneScaler(
            mean=np.zeros(dim, dtype=np.float32),
            std=np.ones(dim, dtype=np.float32),
        )
    )
    ckpt = tmp_path / "ckpt"
    save_checkpoint(model, ckpt, config={"standardize_backbone": True})
    cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    assert cfg["standardize_backbone"] is True
    assert "backbone_scaler" in cfg
    loaded = load_checkpoint(ckpt, device="cpu")
    assert loaded.backbone_scaler is not None
    assert loaded.backbone_scaler.mean.shape == (dim,)


def test_model_kwargs_from_cfg_reads_standardize_flag():
    kw = model_kwargs_from_cfg({"backbone_name": "x", "standardize_backbone": True})
    assert kw["standardize_backbone"] is True
