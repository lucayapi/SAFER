"""Tests découpage batch input, état multi-chunks et reprise run_id."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from annotation.batch_builder import filter_rows_for_batch, write_batch_input_jsonl_chunks
from annotation.batch_client import (
    aggregate_batch_status,
    batch_object_to_state,
    has_pending_download_jobs,
    has_pending_ingest_jobs,
    list_batch_chunks,
    mark_chunks_ingested,
    submit_batch_chunks,
)
from annotation.batch_config import BatchAnnotationConfig
from annotation.cache import append_to_cache, get_output_paths, should_skip_row_for_batch


def _sample_df(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "accident_id": [f"A{i // 3}" for i in range(n)],
            "fact_id": list(range(1, n + 1)),
            "sentence": [f"phrase {i}" for i in range(n)],
            "accident_summary": [f"résumé {i // 3}" for i in range(n)],
        }
    )


def _sample_cfg(tmp_path: Path, **kwargs) -> BatchAnnotationConfig:
    return BatchAnnotationConfig(
        input_csv="x.csv",
        openai_model="gpt-5.4-nano",
        annotation_root=tmp_path,
        run_id="test_run",
        **kwargs,
    )


def test_write_batch_input_jsonl_chunks_splits_by_max_requests(tmp_path):
    cfg = _sample_cfg(tmp_path, max_requests_per_batch=3)
    df = _sample_df(7)
    out = write_batch_input_jsonl_chunks(
        df,
        cfg,
        output_dir=tmp_path,
        base_stem="model__prompt__pass1",
    )
    assert out["n_chunks"] == 3
    assert out["n_requests_written"] == 7
    assert len(out["batch_input_chunks"]) == 3
    assert out["batch_input_chunks"][0]["n_requests"] == 3
    assert out["batch_input_chunks"][-1]["n_requests"] == 1
    for chunk in out["batch_input_chunks"]:
        path = tmp_path / str(chunk["batch_input_path"]).split("\\")[-1].split("/")[-1]
        assert path.is_file()
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == chunk["n_requests"]


def test_write_batch_input_jsonl_single_chunk_uses_default_name(tmp_path):
    cfg = _sample_cfg(tmp_path, max_requests_per_batch=10)
    out = write_batch_input_jsonl_chunks(
        _sample_df(2),
        cfg,
        output_dir=tmp_path,
        base_stem="model__prompt__pass1",
    )
    assert out["n_chunks"] == 1
    assert out["batch_input_path"].endswith("__batch_input.jsonl")


def test_list_batch_chunks_supports_legacy_single_state():
    legacy = {"batch_id": "batch_123", "status": "completed"}
    chunks = list_batch_chunks(legacy)
    assert len(chunks) == 1
    assert chunks[0]["batch_id"] == "batch_123"


def test_aggregate_batch_status_completed_when_all_chunks_done():
    chunks = [
        {"status": "completed", "request_counts": {"total": 2, "completed": 2, "failed": 0}},
        {"status": "completed", "request_counts": {"total": 3, "completed": 3, "failed": 0}},
    ]
    agg = aggregate_batch_status(chunks)
    assert agg["status"] == "completed"
    assert agg["request_counts"]["total"] == 5
    assert agg["request_counts"]["completed"] == 5


def test_filter_rows_for_batch_excludes_cached_ok_rows(tmp_path):
    cfg = _sample_cfg(tmp_path)
    df = _sample_df(3)
    jsonl_path, _, _, _, _ = get_output_paths(
        cfg.outputs_dir,
        model_id=cfg.openai_model,
        prompt_version=cfg.prompt_version,
        artifact_slug=cfg.artifact_slug,
    )
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    append_to_cache(
        jsonl_path,
        {
            "cache_key": "A0||1",
            "pred_ok": True,
            "pred_source": "batch",
        },
    )

    filtered, stats = filter_rows_for_batch(df, cfg)
    assert stats["n_cache_hits"] == 1
    assert stats["n_batch_requests"] == 2
    assert len(filtered) == 2


def test_filter_rows_for_batch_excludes_failed_after_max_retries(tmp_path):
    cfg = _sample_cfg(tmp_path, max_batch_retries=1)
    df = _sample_df(3)
    jsonl_path, _, _, _, _ = get_output_paths(
        cfg.outputs_dir,
        model_id=cfg.openai_model,
        prompt_version=cfg.prompt_version,
        artifact_slug=cfg.artifact_slug,
    )
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    append_to_cache(
        jsonl_path,
        {
            "cache_key": "A0||1",
            "pred_ok": False,
            "pred_error": "PARSE_ERROR",
            "batch_attempts": 1,
        },
    )

    filtered, stats = filter_rows_for_batch(df, cfg)
    assert stats["n_batch_exhausted"] == 1
    assert stats["n_batch_requests"] == 2


def test_should_skip_row_for_batch_legacy_failed_entry():
    cached = {"pred_ok": False, "pred_error": "oops"}
    assert should_skip_row_for_batch(cached, max_batch_retries=1) is True
    assert should_skip_row_for_batch(cached, max_batch_retries=2) is False


def test_submit_batch_chunks_blocks_when_in_progress(tmp_path):
    cfg = _sample_cfg(tmp_path)
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    state_path = cfg.outputs_dir / "batch_state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 2,
                "chunks": [
                    {
                        "batch_id": "batch_running",
                        "status": "in_progress",
                        "request_counts": {"total": 2, "completed": 0, "failed": 0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    chunks = [{"chunk_index": 0, "batch_input_path": str(tmp_path / "in.jsonl"), "n_requests": 1}]
    (tmp_path / "in.jsonl").write_text("{}\n", encoding="utf-8")

    with patch("annotation.batch_client.refresh_batch_state") as refresh:
        refresh.return_value = json.loads(state_path.read_text(encoding="utf-8"))
        result = submit_batch_chunks(cfg, chunks, client=MagicMock())
    assert result["action"] == "wait"


def test_submit_batch_chunks_blocks_until_download_and_ingest(tmp_path):
    cfg = _sample_cfg(tmp_path)
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    state_path = cfg.outputs_dir / "batch_state.json"
    completed_chunk = {
        "batch_id": "batch_done",
        "status": "completed",
        "request_counts": {"total": 2, "completed": 2, "failed": 0},
        "downloaded_at": "2026-07-12T10:00:00+00:00",
    }
    state_path.write_text(
        json.dumps({"version": 2, "chunks": [completed_chunk]}),
        encoding="utf-8",
    )
    chunks = [{"chunk_index": 0, "batch_input_path": str(tmp_path / "in.jsonl"), "n_requests": 1}]
    (tmp_path / "in.jsonl").write_text("{}\n", encoding="utf-8")

    with patch("annotation.batch_client.refresh_batch_state") as refresh:
        refresh.return_value = json.loads(state_path.read_text(encoding="utf-8"))
        result = submit_batch_chunks(cfg, chunks, client=MagicMock())
    assert result["action"] == "ingest_first"

    completed_chunk.pop("downloaded_at")
    state_path.write_text(
        json.dumps({"version": 2, "chunks": [completed_chunk]}),
        encoding="utf-8",
    )
    with patch("annotation.batch_client.refresh_batch_state") as refresh:
        refresh.return_value = json.loads(state_path.read_text(encoding="utf-8"))
        result = submit_batch_chunks(cfg, chunks, client=MagicMock())
    assert result["action"] == "download_first"


def test_batch_object_to_state_preserves_ingested_at():
    batch = MagicMock()
    batch.id = "batch_done"
    batch.input_file_id = "file-1"
    batch.output_file_id = "file-out"
    batch.error_file_id = None
    batch.status = "completed"
    batch.request_counts = None

    previous = {
        "batch_id": "batch_done",
        "ingested_at": "2026-07-12T10:00:00+00:00",
        "downloaded_at": "2026-07-12T09:00:00+00:00",
        "chunk_index": 2,
        "batch_input_path": "/tmp/part003.jsonl",
    }
    state = batch_object_to_state(batch, previous=previous)
    assert state["ingested_at"] == "2026-07-12T10:00:00+00:00"
    assert state["chunk_index"] == 2


def test_mark_chunks_ingested_sets_timestamp(tmp_path):
    cfg = _sample_cfg(tmp_path)
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    state_path = cfg.outputs_dir / "batch_state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 2,
                "chunks": [
                    {
                        "batch_id": "batch_done",
                        "status": "completed",
                        "downloaded_at": "2026-07-12T10:00:00+00:00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert has_pending_download_jobs(json.loads(state_path.read_text())["chunks"]) is False
    assert has_pending_ingest_jobs(json.loads(state_path.read_text())["chunks"]) is True

    mark_chunks_ingested(cfg)
    updated = json.loads(state_path.read_text(encoding="utf-8"))
    assert updated["chunks"][0].get("ingested_at")
    assert has_pending_ingest_jobs(updated["chunks"]) is False
