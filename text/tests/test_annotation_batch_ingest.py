"""Tests ingestion résultats batch annotation."""

from __future__ import annotations

import json

import pandas as pd

from annotation.aggregate import aggregate_outcomes_by_accident
from annotation.batch_ingest import parse_batch_result_record
from annotation.batch_config import BatchAnnotationConfig, align_cfg_for_existing_run
from annotation.runner import results_dataframe_for_merge


def _ok_record(custom_id: str, content: str) -> dict:
    return {
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "body": {
                "choices": [{"message": {"content": content}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "prompt_tokens_details": {"cached_tokens": 80},
                },
            },
        },
        "error": None,
    }


def test_parse_batch_result_record_success():
    content = json.dumps(
        {
            "label": "B",
            "injury_mentioned": "NOT_MENTIONED",
            "hospitalized": "NOT_MENTIONED",
            "fatal": "NOT_MENTIONED",
            "confidence": 0.9,
            "justification": "Contexte utilisé: non. Indice principal: « chute ».",
        },
        ensure_ascii=False,
    )
    res = parse_batch_result_record(_ok_record("A1||F3", content))
    assert res["pred_ok"] is True
    assert res["pred_label"] == "B"
    assert res["pred_context_used"] is False
    assert res["accident_id"] == "A1"
    assert res["fact_id"] == "F3"
    assert res["cached_tokens"] == 80


def test_parse_batch_result_record_api_error():
    res = parse_batch_result_record(
        {
            "custom_id": "A1||F1",
            "error": {"message": "rate limit"},
            "response": None,
        }
    )
    assert res["pred_ok"] is False
    assert "rate limit" in res["pred_error"]


def test_parse_batch_result_record_v13_pass1():
    content = json.dumps(
        {
            "label": "A0",
            "injury_mentioned": "NOT_MENTIONED",
            "hospitalized": "NOT_MENTIONED",
            "fatal": "NOT_MENTIONED",
            "confidence": 0.62,
            "ambiguous": True,
            "context_needed": True,
            "alternative_label": "B",
            "ambiguity_type": "ACTION_INTENT",
            "ambiguity_reason": "doute action",
            "context_used": False,
            "justification": "Contexte utilisé: non. Indice principal: « retiré ».",
        },
        ensure_ascii=False,
    )
    res = parse_batch_result_record(
        _ok_record("A2||F1", content),
        prompt_version="v13_two_pass_ambiguity_context",
        pass_mode="pass1",
    )
    assert res["pred_ok"] is True
    assert res["pred_ambiguous"] is True
    assert res["pred_alternative_label"] == "B"


def test_results_dataframe_for_merge_drops_duplicate_source_columns():
    results = [
        {
            "custom_id": "A1||F1",
            "accident_id": "A1",
            "fact_id": "F1",
            "cache_key": "A1||F1",
            "http_status_code": 200,
            "pred_label": "B",
            "pred_ok": True,
        }
    ]
    merged = results_dataframe_for_merge(results)
    assert "accident_id" not in merged.columns
    assert "fact_id" not in merged.columns
    assert merged["pred_label"].iloc[0] == "B"


def test_merge_with_source_df_keeps_single_accident_id_column():
    content = json.dumps(
        {
            "label": "B",
            "injury_mentioned": "NOT_MENTIONED",
            "hospitalized": "NOT_MENTIONED",
            "fatal": "NOT_MENTIONED",
            "confidence": 0.9,
            "justification": "test",
        },
        ensure_ascii=False,
    )
    results = [parse_batch_result_record(_ok_record("A1||F1", content))]
    source = pd.DataFrame(
        {
            "accident_id": ["A1"],
            "fact_id": ["F1"],
            "unit_text": ["chute"],
        }
    )
    final_df = pd.concat(
        [source.reset_index(drop=True), results_dataframe_for_merge(results)],
        axis=1,
    )
    assert list(final_df.columns).count("accident_id") == 1
    accident_df = aggregate_outcomes_by_accident(final_df)
    assert len(accident_df) == 1
    assert accident_df.iloc[0]["accident_id"] == "A1"


def test_align_cfg_for_existing_run_restores_sampling_and_prompt(tmp_path):
    run_id = "demo__gpt-5.4-mini__v13_two_pass_ambiguity_context__pass1__20260711T000000Z"
    outputs_dir = tmp_path / "outputs" / run_id
    outputs_dir.mkdir(parents=True)
    (outputs_dir / "run_config.json").write_text(
        json.dumps(
            {
                "input_csv": "btp_sentence_accidents.csv",
                "output_basename": "btp_v13_pass1",
                "openai_model": "gpt-5.4-mini",
                "prompt_version": "v13_two_pass_ambiguity_context",
                "pass_mode": "pass1",
                "n_accidents": 10,
                "units_per_accident": "all",
                "accident_sample_seed": 42,
                "run_id": run_id,
            }
        ),
        encoding="utf-8",
    )

    wrong_cfg = BatchAnnotationConfig(
        input_csv="btp_sentence_accidents.csv",
        output_basename="btp_v10_batch",
        run_id=run_id,
        annotation_root=tmp_path,
    )
    aligned = align_cfg_for_existing_run(wrong_cfg)

    assert aligned.prompt_version == "v13_two_pass_ambiguity_context"
    assert aligned.n_accidents == 10
    assert aligned.output_basename == "btp_v13_pass1"
