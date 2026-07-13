"""Tests cache annotation."""

from __future__ import annotations

import json

import pandas as pd

from annotation.cache import append_to_cache, get_output_paths, load_cache, make_cache_key
from annotation.export_io import ANNOTATION_TABLE_SUFFIX


def test_make_cache_key_with_fact_id():
    row = pd.Series({"accident_id": "A1", "fact_id": "3"})
    assert make_cache_key(row, row_idx=0) == "A1||3"


def test_make_cache_key_fallback_rowidx():
    row = pd.Series({"accident_id": "A1"})
    assert make_cache_key(row, row_idx=7) == "A1||ROWIDX_7"


def test_get_output_paths_use_xlsx(tmp_path):
    paths = get_output_paths(
        tmp_path,
        model_id="gpt-5-nano",
        prompt_version="v10_macro_labels_independent_outcomes",
    )
    assert paths[0].suffix == ".jsonl"
    for path in paths[1:]:
        assert path.suffix == ANNOTATION_TABLE_SUFFIX


def test_load_and_append_cache_roundtrip(tmp_path):
    path = tmp_path / "cache.jsonl"
    record = {"cache_key": "A||1", "pred_ok": True, "pred_label": "B"}
    append_to_cache(path, record)
    append_to_cache(path, {"cache_key": "A||2", "pred_ok": False})
    cache = load_cache(path)
    assert len(cache) == 2
    assert cache["A||1"]["pred_label"] == "B"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["cache_key"] == "A||1"
