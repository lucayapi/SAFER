"""Tests construction requêtes batch annotation."""

from __future__ import annotations

import pandas as pd

from annotation.batch_config import BatchAnnotationConfig
from annotation.batch_request import build_batch_request_line, build_chat_completion_body
from annotation.cache import make_cache_key


def test_build_chat_completion_body_uses_json_format():
    cfg = BatchAnnotationConfig(
        input_csv="x.csv",
        openai_model="gpt-5-mini",
        reasoning_effort="medium",
        max_output_tokens=4000,
    )
    row = {
        "accident_id": "A1",
        "fact_id": "3",
        "sentence": "Il chute.",
        "accident_summary": "Chute sur chantier",
    }
    body = build_chat_completion_body(row, cfg, for_batch=True)
    assert body["model"] == "gpt-5-mini"
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_completion_tokens"] >= 4000
    assert body["reasoning_effort"] == "medium"
    assert "extra_body" not in body
    assert body["prompt_cache_key"].startswith("safer-annotation:")


def test_build_batch_request_line_structure():
    cfg = BatchAnnotationConfig(
        input_csv="x.csv",
        openai_model="gpt-5-mini",
        reasoning_effort="medium",
    )
    row = pd.Series(
        {
            "accident_id": "ACC1",
            "fact_id": "F2",
            "sentence": "phrase",
            "accident_summary": "résumé",
        }
    )
    line = build_batch_request_line(
        custom_id=make_cache_key(row, row_idx=0),
        row=row,
        cfg=cfg,
    )
    assert line["custom_id"] == "ACC1||F2"
    assert line["method"] == "POST"
    assert line["url"] == "/v1/chat/completions"
    assert "messages" in line["body"]
