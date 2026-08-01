"""Tests logs supervised_macro_ft."""

from __future__ import annotations

import torch

from supervised_macro_ft.run_logging import log_run_complete, resolve_effective_use_amp


def test_resolve_effective_use_amp_explicit_false():
    enabled, reason = resolve_effective_use_amp(
        {"use_amp": False},
        backbone_trainable=True,
        device=torch.device("cuda"),
    )
    assert enabled is False
    assert "use_amp" in reason


def test_resolve_effective_use_amp_auto_cuda():
    enabled, reason = resolve_effective_use_amp(
        {},
        backbone_trainable=True,
        device=torch.device("cuda"),
    )
    assert enabled is True
    assert "auto" in reason


def test_resolve_effective_use_amp_frozen_cpu():
    enabled, _ = resolve_effective_use_amp(
        {},
        backbone_trainable=False,
        device=torch.device("cpu"),
    )
    assert enabled is False


def test_log_run_complete_without_checkpoint(caplog):
    log_run_complete(output_dir="output/run", checkpoint_dir=None)

    assert "checkpoint=not saved" in caplog.text
