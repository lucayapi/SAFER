"""GOLD export provenance and five-class input checks, without a model download."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from safer_core.test_corpus import TestCorpusSpec as CorpusSpec
from scgm_text.data_metadata import load_filtered_metadata
from scgm_text.dataset_text_embeddings import merge_metadata_with_embeddings
from scripts.export_gold_embeddings import EXPECTED_GOLD_SHA, export_gold, validate_export


def test_gold_export_atomic_resume_and_stale_rejection(tmp_path: Path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "export_embeddings.yaml").write_text(
        yaml.safe_dump({
            "backbone_name": "Qwen/Qwen3-Embedding-0.6B", "backbone_id": "qwen3",
            "output_root": "embeddings/gold_v3", "text_col": "sentence",
        }), encoding="utf-8",
    )
    (tmp_path / "dataset").mkdir()
    data = tmp_path / "dataset" / "data_btp.csv"
    pd.DataFrame([
        {"accident_id": "a", "doc_id": 1, "sentence": "test", "pred_label": "AUTRE", "pred_ok": True},
        {"accident_id": "b", "doc_id": 2, "sentence": "fact", "pred_label": "A0", "pred_ok": True},
    ]).to_csv(data, index=False)
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    (data.parent / "gold_v3_manifest.json").write_text(json.dumps({
        "gold_version": "narrative_v3", "snapshot_sha256": EXPECTED_GOLD_SHA,
        "corpora": {"btp": {"file": data.name, "sha256": digest, "units": 2}},
    }), encoding="utf-8")
    spec = CorpusSpec("btp", "BTP", data, tmp_path / "missing.csv")

    def fake_export(cfg, dataset, path, **kwargs):
        pd.DataFrame({"doc_id": [1, 2], "dim_0001": [0.1, 0.2]}).to_csv(path, index=False)

    with patch("scripts.export_gold_embeddings.resolve_test_corpus", return_value=spec), \
         patch("scripts.export_gold_embeddings.export_text_embeddings", side_effect=fake_export) as encode:
        dest = export_gold("qwen3", "btp", root=tmp_path)
        assert encode.call_count == 1
        assert validate_export(dest, [1, 2]) == 1
        assert dest.with_suffix(".manifest.json").is_file()
        metadata = load_filtered_metadata(str(data))
        merged, _ = merge_metadata_with_embeddings(metadata, str(dest))
        assert len(merged) == 2
        metadata.attrs["gold_dataset_sha256"] = "wrong"
        with pytest.raises(ValueError, match="provenance differs"):
            merge_metadata_with_embeddings(metadata, str(dest))
        export_gold("qwen3", "btp", root=tmp_path)
        assert encode.call_count == 1
        data.write_text(data.read_text() + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="GOLD dataset/manifest mismatch"):
            export_gold("qwen3", "btp", root=tmp_path)


def test_embedding_ids_must_match_order(tmp_path: Path):
    path = tmp_path / "vectors.csv"
    path.write_text("doc_id,dim_0001\n2,0.1\n1,0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="IDs/order mismatch"):
        validate_export(path, [1, 2])
