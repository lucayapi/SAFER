"""Tests registre corpus de test."""

from __future__ import annotations

import pytest

from safer_core.paths import TEXT_ROOT
from safer_core.test_corpus import (
    default_test_corpus_id,
    list_test_corpus_ids,
    macro_transfer_output_dir,
    method_test_results_dir,
    raw_embedding_test_dir,
    resolve_test_corpus,
)


def test_default_corpus_is_metallurgie():
    assert default_test_corpus_id() == "metallurgie"


def test_list_includes_metallurgie():
    assert "metallurgie" in list_test_corpus_ids()


def test_resolve_metallurgie_paths():
    spec = resolve_test_corpus("metallurgie", anchor=TEXT_ROOT)
    assert spec.id == "metallurgie"
    assert spec.data_csv.name == "data_metallurgie.csv"
    assert spec.emb_csv.name == "Qwen3-Embedding-0.6B__metallurgie.csv"
    assert spec.emb_csv.is_file()
    assert spec.data_csv.is_absolute()


def test_unknown_corpus_raises():
    with pytest.raises(KeyError, match="inconnu"):
        resolve_test_corpus("nonexistent_corpus_xyz", anchor=TEXT_ROOT)


def test_bn_results_dir():
    from safer_core.test_corpus import bn_results_dir

    p = bn_results_dir("metallurgie", anchor=TEXT_ROOT)
    assert p.parts[-2:] == ("metallurgie", "bn_results")


def test_bn_staging_dir_alias():
    from safer_core.test_corpus import bn_results_dir, bn_staging_dir

    p = bn_staging_dir("metallurgie", anchor=TEXT_ROOT)
    assert p == bn_results_dir("metallurgie", anchor=TEXT_ROOT)


def test_output_test_layout():
    mt = macro_transfer_output_dir("scgm_text", "metallurgie", anchor=TEXT_ROOT)
    assert mt.parts[-3:] == ("metallurgie", "macro_transfer", "scgm_text")
    m = method_test_results_dir("softtriple", "metallurgie", anchor=TEXT_ROOT)
    assert m.name == "softtriple"
    assert m.parent.name == "metallurgie"
    raw = raw_embedding_test_dir("metallurgie", anchor=TEXT_ROOT)
    assert raw.name == "raw_embedding"
