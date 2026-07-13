"""Tests registre corpus de test."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from safer_core.paths import TEXT_ROOT
from safer_core.test_corpus import (
    conventional_test_paths,
    default_test_corpus_id,
    list_test_corpus_ids,
    macro_transfer_output_dir,
    method_test_results_dir,
    raw_embedding_test_dir,
    resolve_projected_embeddings_paths,
    resolve_test_corpus,
)


def test_default_corpus_is_metallurgie():
    assert default_test_corpus_id() == "metallurgie"


def test_list_includes_metallurgie_and_btp():
    ids = list_test_corpus_ids()
    assert "metallurgie" in ids
    assert "btp" in ids
    assert "caou" in ids


def test_resolve_metallurgie_paths():
    spec = resolve_test_corpus("metallurgie", anchor=TEXT_ROOT)
    assert spec.id == "metallurgie"
    assert spec.data_csv.name == "data_metallurgie.csv"
    assert spec.emb_csv.name == "Qwen3-Embedding-0.6B_metallurgie.csv"
    assert spec.data_csv.is_absolute()
    assert "test" not in str(spec.data_csv)


def test_conventional_paths_flat():
    data_rel, emb_rel = conventional_test_paths("caou")
    assert data_rel.endswith("data_caou.csv")
    assert emb_rel.endswith("Qwen3-Embedding-0.6B_caou.csv")


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


def test_resolve_projected_embeddings_paths(tmp_path):
    emb_dir = tmp_path / "output" / "supcon" / "embeddings"
    emb_dir.mkdir(parents=True)
    npy = emb_dir / "projected_btp.npy"
    meta = emb_dir / "projected_btp_metadata.csv"
    npy.write_bytes(b"")
    meta.write_text("pred_label\nA0\n", encoding="utf-8")

    with patch("safer_core.test_corpus.method_btp_results_dir", return_value=tmp_path / "output" / "supcon"):
        pair = resolve_projected_embeddings_paths("supcon", "btp", anchor=tmp_path)
    assert pair is not None
    assert pair[0].name == "projected_btp.npy"
    assert pair[1].name == "projected_btp_metadata.csv"
