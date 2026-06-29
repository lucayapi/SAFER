"""Tests checkpoint supervised_macro_ft."""

from __future__ import annotations

from pathlib import Path

import torch

from supervised_macro_ft.checkpoint_io import load_checkpoint, read_checkpoint_config, save_checkpoint
from supervised_macro_ft.model import SupervisedMacroModel


def test_checkpoint_roundtrip_ln_gelu(tmp_path: Path):
    model = SupervisedMacroModel(
        backbone_name="__test_dummy__",
        num_classes=4,
        backbone_trainable=False,
        projection="ln_gelu",
        hiddim=32,
        proj_hidden=48,
        dropout=0.0,
    )
    save_checkpoint(
        model,
        tmp_path / "ln_model",
        config={
            "backbone_name": "__test_dummy__",
            "n_classes": 4,
            "projection": "ln_gelu",
            "hiddim": 32,
            "proj_hidden": 48,
        },
    )
    cfg = read_checkpoint_config(tmp_path / "ln_model")
    assert cfg["projection"] == "ln_gelu"
    assert cfg["proj_hidden"] == 48
    loaded = load_checkpoint(tmp_path / "ln_model", device="cpu")
    x = torch.randint(0, 20, (2, 5))
    m = torch.ones(2, 5, dtype=torch.long)
    with torch.no_grad():
        z1 = model.encode(x, m)
        z2 = loaded.encode(x, m)
    assert torch.allclose(z1, z2, atol=1e-5)


def test_checkpoint_roundtrip_with_projector(tmp_path: Path):
    model = SupervisedMacroModel(
        backbone_name="__test_dummy__",
        num_classes=4,
        backbone_trainable=False,
        projection="mlp",
        hiddim=32,
        dropout=0.0,
    )
    save_checkpoint(
        model,
        tmp_path / "best_model",
        config={
            "backbone_name": "__test_dummy__",
            "n_classes": 4,
            "projection": "mlp",
            "hiddim": 32,
        },
    )
    cfg = read_checkpoint_config(tmp_path / "best_model")
    assert cfg["projection"] == "mlp"
    assert (tmp_path / "best_model" / "projector.pt").is_file()
    loaded = load_checkpoint(tmp_path / "best_model", device="cpu")
    x = torch.randint(0, 20, (2, 5))
    m = torch.ones(2, 5, dtype=torch.long)
    with torch.no_grad():
        p1, _ = model.predict_proba(x, m)
        p2, _ = loaded.predict_proba(x, m)
        z1 = model.encode(x, m)
        z2 = loaded.encode(x, m)
    assert torch.allclose(p1, p2, atol=1e-5)
    assert torch.allclose(z1, z2, atol=1e-5)


def test_checkpoint_legacy_without_projector(tmp_path: Path):
    model = SupervisedMacroModel(
        backbone_name="__test_dummy__",
        num_classes=4,
        backbone_trainable=False,
        projection=None,
    )
    save_checkpoint(
        model,
        tmp_path / "legacy_model",
        config={"backbone_name": "__test_dummy__", "n_classes": 4},
    )
    assert not (tmp_path / "legacy_model" / "projector.pt").exists()
    loaded = load_checkpoint(tmp_path / "legacy_model", device="cpu")
    x = torch.randint(0, 20, (2, 5))
    m = torch.ones(2, 5, dtype=torch.long)
    with torch.no_grad():
        p1, _ = model.predict_proba(x, m)
        p2, _ = loaded.predict_proba(x, m)
    assert torch.allclose(p1, p2, atol=1e-5)
