"""Tests checkpoint supervised_macro_ft."""

from __future__ import annotations

from pathlib import Path

import torch

from supervised_macro_ft.checkpoint_io import load_checkpoint, read_checkpoint_config, save_checkpoint
from supervised_macro_ft.model import SupervisedMacroModel


def test_checkpoint_roundtrip(tmp_path: Path):
    model = SupervisedMacroModel(
        backbone_name="__test_dummy__",
        num_classes=4,
        backbone_trainable=False,
    )
    save_checkpoint(model, tmp_path / "best_model", config={"backbone_name": "__test_dummy__", "n_classes": 4})
    cfg = read_checkpoint_config(tmp_path / "best_model")
    assert cfg["backbone_name"] == "__test_dummy__"
    loaded = load_checkpoint(tmp_path / "best_model", device="cpu")
    x = torch.randint(0, 20, (2, 5))
    m = torch.ones(2, 5, dtype=torch.long)
    with torch.no_grad():
        p1, _ = model.predict_proba(x, m)
        p2, _ = loaded.predict_proba(x, m)
    assert torch.allclose(p1, p2, atol=1e-5)
