"""Tests résolution FSP (checkpoint, output_dir)."""

from __future__ import annotations

from pathlib import Path

import pytest

from macro_transfer.fsp_config import (
    FSP_SOFTTRIPLE_NATIVE_METHOD,
    fsp_encoder_method,
    fsp_output_method_key,
    normalize_fsp_base_method,
    resolve_fsp_checkpoint,
    resolve_fsp_method_display_name,
    resolve_fsp_output_dir,
    validate_fsp_base_method,
    validate_fsp_method,
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


def test_resolve_fsp_output_dir_legacy_scgm(tmp_path: Path):
    corpus = "metallurgie"
    legacy = (
        tmp_path
        / "output_test"
        / corpus
        / "macro_transfer"
        / "frozen_source_prototypes"
        / "scgm"
        / "transfer"
    )
    legacy.mkdir(parents=True)
    (legacy / "metrics.json").write_text('{"balanced_accuracy": 0.42}', encoding="utf-8")

    p = resolve_fsp_output_dir(corpus, "scgm_text", anchor=tmp_path)
    assert p.name == "scgm"
    assert (p / "transfer" / "metrics.json").is_file()


def test_resolve_fsp_output_dir_prefers_canonical_over_legacy(tmp_path: Path):
    corpus = "metallurgie"
    root = tmp_path / "output_test" / corpus / "macro_transfer" / "frozen_source_prototypes"
    for method, score in (("scgm_text", 0.9), ("scgm", 0.1)):
        transfer = root / method / "transfer"
        transfer.mkdir(parents=True)
        (transfer / "metrics.json").write_text(
            f'{{"balanced_accuracy": {score}}}', encoding="utf-8"
        )

    p = resolve_fsp_output_dir(corpus, "scgm_text", anchor=tmp_path)
    assert p.name == "scgm_text"


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


def test_softtriple_native_method_resolution():
    assert validate_fsp_method("softtriple_native") == FSP_SOFTTRIPLE_NATIVE_METHOD
    assert fsp_encoder_method("softtriple_native") == "softtriple"
    assert fsp_output_method_key("softtriple_native") == "frozen_source_prototypes/softtriple_native"
    assert "centres natifs" in resolve_fsp_method_display_name("softtriple_native").lower()


def test_resolve_fsp_checkpoint_softtriple_native_falls_back_to_softtriple():
    ckpt = resolve_fsp_checkpoint(
        FSP_SOFTTRIPLE_NATIVE_METHOD,
        {},
        {"softtriple": "output/softtriple/checkpoints/best_model"},
    )
    assert ckpt == "output/softtriple/checkpoints/best_model"


def test_resolve_fsp_output_dir_softtriple_native():
    p = resolve_fsp_output_dir("metallurgie", FSP_SOFTTRIPLE_NATIVE_METHOD, anchor=TEXT_ROOT)
    assert p.name == "softtriple_native"
