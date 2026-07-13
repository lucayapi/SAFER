"""Tests préparation fichier batch input."""

from __future__ import annotations

import json

import pandas as pd

from annotation.batch_builder import filter_rows_for_batch, write_batch_input_jsonl
from annotation.batch_config import BatchAnnotationConfig
from annotation.cache import append_to_cache, get_output_paths


def test_filter_rows_for_batch_excludes_cache_hits(tmp_path):
    cfg = BatchAnnotationConfig(
        input_csv="x.csv",
        output_basename="batch_test",
        run_id="test_run",
        annotation_root=tmp_path / "annotation",
        skip_cache=False,
    )
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path, _, _, _, _ = get_output_paths(
        cfg.outputs_dir,
        model_id=cfg.openai_model,
        prompt_version=cfg.prompt_version,
        artifact_slug=cfg.artifact_slug,
    )
    append_to_cache(
        jsonl_path,
        {
            "cache_key": "A1||1",
            "pred_ok": True,
            "pred_label": "B",
        },
    )

    df = pd.DataFrame(
        [
            {"accident_id": "A1", "fact_id": 1, "sentence": "s1", "accident_summary": "r1"},
            {"accident_id": "A1", "fact_id": 2, "sentence": "s2", "accident_summary": "r1"},
        ]
    )
    filtered, stats = filter_rows_for_batch(df, cfg)
    assert len(filtered) == 1
    assert stats["n_cache_hits"] == 1
    assert stats["n_batch_requests"] == 1


def test_write_batch_input_jsonl_format(tmp_path):
    cfg = BatchAnnotationConfig(
        input_csv="x.csv",
        openai_model="gpt-5-nano",
        annotation_root=tmp_path,
    )
    df = pd.DataFrame(
        [
            {
                "accident_id": "A1",
                "fact_id": 9,
                "sentence": "chute",
                "accident_summary": "résumé",
            }
        ]
    )
    out = tmp_path / "batch_input.jsonl"
    write_batch_input_jsonl(df, cfg, output_path=out)
    line = json.loads(out.read_text(encoding="utf-8").strip())
    assert line["custom_id"] == "A1||9"
    assert line["url"] == "/v1/chat/completions"
    assert line["body"]["model"] == "gpt-5-nano"
