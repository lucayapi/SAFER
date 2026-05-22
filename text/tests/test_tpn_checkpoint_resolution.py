"""Tests résolution checkpoint TPN (priorité CLI / base_method)."""

from __future__ import annotations

import pytest

from macro_transfer.tpn_encode import resolve_tpn_checkpoint


def test_override_ignores_method_checkpoint_softtriple():
    ckpt = resolve_tpn_checkpoint(
        "supcon",
        {"checkpoint": "output/softtriple/checkpoints/best_model"},
        {"supcon": "output/supcon/checkpoints/best_model"},
        base_method_overridden=True,
    )
    assert "supcon" in ckpt
    assert "softtriple" not in ckpt


def test_explicit_checkpoint_priority():
    ckpt = resolve_tpn_checkpoint(
        "supcon",
        {"checkpoint": "output/softtriple/checkpoints/best_model"},
        {"supcon": "output/supcon/checkpoints/best_model"},
        explicit_checkpoint="output/custom/ckpt",
        base_method_overridden=True,
    )
    assert ckpt == "output/custom/ckpt"


def test_without_override_uses_method_checkpoint():
    ckpt = resolve_tpn_checkpoint(
        "softtriple",
        {"checkpoint": "output/softtriple/checkpoints/best_model"},
        {"supcon": "output/supcon/checkpoints/best_model"},
        base_method_overridden=False,
    )
    assert "softtriple" in ckpt


def test_without_override_falls_back_to_block():
    ckpt = resolve_tpn_checkpoint(
        "batch_triplet",
        {},
        {"batch_triplet": "output/batch_triplet/checkpoints/best_model"},
        base_method_overridden=False,
    )
    assert "batch_triplet" in ckpt


def test_softtriple_from_block_only():
    ckpt = resolve_tpn_checkpoint(
        "softtriple",
        {},
        {"softtriple": "output/softtriple/checkpoints/best_model"},
    )
    assert "softtriple" in ckpt


def test_override_missing_block_raises():
    with pytest.raises(ValueError, match="Checkpoint manquant"):
        resolve_tpn_checkpoint(
            "supcon",
            {"checkpoint": "output/softtriple/checkpoints/best_model"},
            {},
            base_method_overridden=True,
        )
