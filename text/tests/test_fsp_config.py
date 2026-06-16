"""Tests résolution FSP (checkpoint, output_dir)."""

from __future__ import annotations

from pathlib import Path

import pytest

from macro_transfer.fsp_config import (
    fsp_output_method_key,
    normalize_fsp_base_method,
    resolve_fsp_checkpoint,
    resolve_fsp_output_dir,
    validate_fsp_base_method,
)

TEXT_ROOT = Path(__file__).resolve().parents[1]


def test_fsp_output_method_key():
    assert fsp_output_method_key("scgm_text") == "frozen_source_prototypes/scgm_text"
    assert fsp_output_method_key("raw_embedding") == "frozen_source_prototypes/raw_embedding"


def test_normalize_fsp_aliases():
    assert normalize_fsp_base_method("scgm") == "scgm_text"
    assert normalize_fsp_base_method("raw") == "raw_embedding"
    assert normalize_fsp_base_method("frozen_source_prototypes/scgm") == "scgm_text"


def test_resolve_fsp_output_dir_default():
    p = resolve_fsp_output_dir("metallurgie", "softtriple", anchor=TEXT_ROOT)
    assert p.name == "softtriple"
    assert "frozen_source_prototypes" in str(p)
    assert "metallurgie" in str(p)


def test_resolve_fsp_checkpoint_from_block():
    ckpt = resolve_fsp_checkpoint(
        "softtriple",
        {},
        {"softtriple": "output/softtriple/checkpoints/best_model"},
    )
    assert ckpt == "output/softtriple/checkpoints/best_model"


def test_resolve_fsp_checkpoint_raw_is_none():
    assert resolve_fsp_checkpoint("raw_embedding", {}, {}) is None


def test_validate_fsp_base_method_rejects_unknown():
    with pytest.raises(ValueError, match="non supporté"):
        validate_fsp_base_method("tpn_full_scgm_text")
