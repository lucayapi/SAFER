"""Tests export_corpus_embeddings (config + registre, sans GPU)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from safer_core.test_corpus import (
    backbone_short_id,
    conventional_test_paths,
    list_test_corpus_ids,
    resolve_test_corpus,
)

TEXT_ROOT = Path(__file__).resolve().parents[1]


def test_registry_lists_three_corpora():
    ids = list_test_corpus_ids()
    assert ids == ["btp", "caou", "metallurgie"]


def test_conventional_flat_paths():
    data_rel, emb_rel = conventional_test_paths("metallurgie")
    assert data_rel == "dataset/data_metallurgie.csv"
    assert emb_rel == "embeddings/Qwen3-Embedding-0.6B_metallurgie.csv"


def test_resolve_metallurgie_flat_paths():
    spec = resolve_test_corpus("metallurgie", anchor=TEXT_ROOT)
    assert spec.id == "metallurgie"
    assert spec.data_csv.name == "data_metallurgie.csv"
    assert spec.emb_csv.name == "Qwen3-Embedding-0.6B_metallurgie.csv"
    assert "dataset" in str(spec.data_csv)
    assert "test" not in spec.data_csv.parts[-2:]
    assert spec.data_csv.is_absolute()


def test_resolve_btp_paths():
    spec = resolve_test_corpus("btp", anchor=TEXT_ROOT)
    assert spec.data_csv.name == "data_btp.csv"
    assert spec.emb_csv.name == "Qwen3-Embedding-0.6B_btp.csv"


def test_backbone_short_id():
    assert backbone_short_id("Qwen/Qwen3-Embedding-0.6B") == "Qwen3-Embedding-0.6B"


def test_load_export_config():
    from scripts.export_corpus_embeddings import load_export_config, resolve_corpora_ids

    cfg = load_export_config(TEXT_ROOT / "configs" / "export_embeddings.yaml")
    assert cfg["text_col"] == "sentence"
    assert cfg["backbone_name"] == "Qwen/Qwen3-Embedding-0.6B"
    assert resolve_corpora_ids(cfg) == ["btp", "metallurgie", "caou"]
    assert resolve_corpora_ids(cfg, corpus="caou") == ["caou"]
    assert resolve_corpora_ids(cfg, all_corpora=True) == ["btp", "caou", "metallurgie"]


def test_skip_existing_without_force(tmp_path, monkeypatch):
    from scripts.export_corpus_embeddings import export_corpus, load_export_config

    registry = {
        "default": "mini",
        "corpora": {
            "mini": {
                "display_name": "Mini",
                "data_csv": "data.csv",
                "emb_csv": "emb.csv",
            }
        },
    }
    reg_path = tmp_path / "test_corpora.yaml"
    reg_path.write_text(yaml.safe_dump(registry), encoding="utf-8")

    data_csv = tmp_path / "data.csv"
    data_csv.write_text(
        "accident_id,fact_id,sentence,pred_label,pred_ok\n"
        "a1,1,phrase test,A0,true\n",
        encoding="utf-8",
    )
    emb_csv = tmp_path / "emb.csv"
    emb_csv.write_text("doc_id,dim_0001\n0,0.1\n", encoding="utf-8")

    cfg = load_export_config(TEXT_ROOT / "configs" / "export_embeddings.yaml")
    cfg["skip_existing"] = True

    with patch("scripts.export_corpus_embeddings.resolve_test_corpus") as mock_resolve:
        from safer_core.test_corpus import TestCorpusSpec

        mock_resolve.return_value = TestCorpusSpec(
            id="mini",
            display_name="Mini",
            data_csv=data_csv,
            emb_csv=emb_csv,
        )
        with patch("scripts.export_corpus_embeddings.export_text_embeddings") as mock_export:
            export_corpus("mini", cfg, force=False, anchor=tmp_path)
            mock_export.assert_not_called()


def test_force_reencode_calls_export(tmp_path):
    from scripts.export_corpus_embeddings import export_corpus, load_export_config

    data_csv = tmp_path / "data.csv"
    data_csv.write_text(
        "accident_id,fact_id,sentence,pred_label,pred_ok\n"
        "a1,1,phrase test,A0,true\n",
        encoding="utf-8",
    )
    emb_csv = tmp_path / "emb.csv"
    emb_csv.write_text("doc_id,dim_0001\n0,0.1\n", encoding="utf-8")

    cfg = load_export_config(TEXT_ROOT / "configs" / "export_embeddings.yaml")

    with patch("scripts.export_corpus_embeddings.resolve_test_corpus") as mock_resolve:
        from safer_core.test_corpus import TestCorpusSpec

        mock_resolve.return_value = TestCorpusSpec(
            id="mini",
            display_name="Mini",
            data_csv=data_csv,
            emb_csv=emb_csv,
        )
        with patch("scripts.export_corpus_embeddings.prepare_text_dataset") as mock_prep:
            mock_prep.return_value = type("DS", (), {"__len__": lambda self: 1})()
            with patch("scripts.export_corpus_embeddings.export_text_embeddings") as mock_export:
                export_corpus("mini", cfg, force=True, anchor=tmp_path)
                mock_export.assert_called_once()


def test_resolve_does_not_require_emb_when_disabled(tmp_path):
    registry = {
        "default": "mini",
        "corpora": {
            "mini": {
                "display_name": "Mini",
                "data_csv": "data.csv",
                "emb_csv": "emb.csv",
            }
        },
    }
    reg_path = tmp_path / "test_corpora.yaml"
    reg_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    data_csv = tmp_path / "data.csv"
    data_csv.write_text(
        "accident_id,sentence,pred_label,pred_ok\na,s,A0,true\n",
        encoding="utf-8",
    )

    spec = resolve_test_corpus(
        "mini",
        registry_path=reg_path,
        anchor=tmp_path,
        require_files=True,
        require_emb_csv=False,
    )
    assert spec.data_csv.is_file()
    assert not spec.emb_csv.is_file()
