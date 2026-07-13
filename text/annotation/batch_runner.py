"""Export final après ingestion d'un batch OpenAI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from annotation.aggregate import aggregate_outcomes_by_accident, summarize_predictions
from annotation.batch_builder import load_input_dataframe
from annotation.batch_config import BatchAnnotationConfig, align_cfg_for_existing_run
from annotation.batch_ingest import parse_batch_output_jsonl, results_by_cache_key
from annotation.cache import append_to_cache, batch_attempt_count, get_batch_paths, get_output_paths, load_cache, make_cache_key
from annotation.export_io import (
    attach_accident_summary_column,
    reorder_annotation_output_columns,
    save_annotation_table,
)
from annotation.prompts import empty_prediction_fields
from annotation.runner import (
    _cache_record_from_result,
    _result_from_cache,
    prepare_annotation_frame,
    results_dataframe_for_merge,
)
from annotation.batch_client import collect_batch_output_paths, mark_chunks_ingested
from annotation.sampling import sampling_stats


def _merge_results_for_frame(
    df: pd.DataFrame,
    cfg: BatchAnnotationConfig,
    parsed_by_key: Dict[str, Dict[str, Any]],
    cache: Dict[str, Dict[str, Any]],
    *,
    jsonl_path,
) -> tuple[List[Dict[str, Any]], dict[str, int]]:
    results: List[Dict[str, Any]] = []
    stats = {"cache_hits": 0, "batch_new": 0, "batch_errors": 0}

    for idx, row in df.iterrows():
        cache_key = make_cache_key(row, row_idx=idx)
        cached = cache.get(cache_key)
        if cached and cached.get("pred_ok"):
            res = _result_from_cache(cached)
            stats["cache_hits"] += 1
        elif cache_key in parsed_by_key:
            res = dict(parsed_by_key[cache_key])
            if res.get("pred_ok"):
                stats["batch_new"] += 1
            else:
                stats["batch_errors"] += 1
            record = _cache_record_from_result(
                cache_key=cache_key,
                cfg=cfg,
                row=row,
                res=res,
            )
            record["batch_attempts"] = batch_attempt_count(cached) + 1
            if not (cached and cached.get("pred_ok")):
                append_to_cache(jsonl_path, record)
                cache[cache_key] = record
        elif cached:
            res = _result_from_cache(cached)
            stats["batch_errors"] += 1
        else:
            res = {
                **empty_prediction_fields(cfg.prompt_version),
                "pred_error": None,
                "pred_source": "pending",
                "pred_raw": None,
                "usage_tokens": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "cached_tokens": None,
            }
        results.append(res)
    return results, stats


def ingest_batch_results(cfg: BatchAnnotationConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Lit batch_output.jsonl, met à jour le cache et exporte les XLSX."""
    cfg = align_cfg_for_existing_run(cfg)
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    _, output_path, error_path, _ = get_batch_paths(
        cfg.outputs_dir,
        model_id=cfg.openai_model,
        prompt_version=cfg.prompt_version,
        artifact_slug=cfg.artifact_slug,
    )
    output_paths = collect_batch_output_paths(cfg)
    if not output_paths:
        raise FileNotFoundError(
            f"Aucun fichier batch output dans {cfg.outputs_dir}. Lancez d'abord download."
        )

    extra = [error_path] if error_path.is_file() else None
    parsed: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    for path in output_paths:
        for record in parse_batch_output_jsonl(
            path,
            extra_paths=None,
            prompt_version=cfg.prompt_version,
            pass_mode=cfg.pass_mode,
        ):
            key = str(record.get("cache_key") or record.get("custom_id") or "")
            if key and key in seen_keys:
                continue
            if key:
                seen_keys.add(key)
            parsed.append(record)
    if extra:
        for record in parse_batch_output_jsonl(
            Path(extra[0]),
            prompt_version=cfg.prompt_version,
            pass_mode=cfg.pass_mode,
        ):
            key = str(record.get("cache_key") or record.get("custom_id") or "")
            if key and key in seen_keys:
                continue
            if key:
                seen_keys.add(key)
            parsed.append(record)
    parsed_by_key = results_by_cache_key(parsed)

    df_raw = load_input_dataframe(cfg)
    df_work = prepare_annotation_frame(cfg, df_raw)

    jsonl_path, _, annotated_xlsx_path, summary_xlsx_path, accident_xlsx_path = get_output_paths(
        cfg.outputs_dir,
        model_id=cfg.openai_model,
        prompt_version=cfg.prompt_version,
        artifact_slug=cfg.artifact_slug,
    )
    cache = {} if cfg.skip_cache else load_cache(jsonl_path)

    results, ingest_stats = _merge_results_for_frame(
        df_work,
        cfg,
        parsed_by_key,
        cache,
        jsonl_path=jsonl_path,
    )

    final_df = pd.concat(
        [df_work.reset_index(drop=True), results_dataframe_for_merge(results)],
        axis=1,
    )
    final_df = reorder_annotation_output_columns(
        attach_accident_summary_column(final_df, summary_col=cfg.summary_col)
    )
    save_annotation_table(final_df, annotated_xlsx_path)

    summary_df = summarize_predictions(final_df)
    save_annotation_table(summary_df, summary_xlsx_path)
    accident_df = aggregate_outcomes_by_accident(final_df, summary_col=cfg.summary_col)
    save_annotation_table(accident_df, accident_xlsx_path)

    total_tokens = sum(int(r["usage_tokens"]) for r in results if r.get("usage_tokens"))
    total_prompt_tokens = sum(int(r["prompt_tokens"]) for r in results if r.get("prompt_tokens"))
    total_cached_tokens = sum(int(r["cached_tokens"]) for r in results if r.get("cached_tokens"))
    prompt_cache_hit_rate = None
    if total_prompt_tokens > 0 and total_cached_tokens > 0:
        prompt_cache_hit_rate = float(total_cached_tokens) / float(total_prompt_tokens)

    meta = {
        "jsonl_path": str(jsonl_path),
        "annotated_xlsx_path": str(annotated_xlsx_path),
        "summary_xlsx_path": str(summary_xlsx_path),
        "accident_xlsx_path": str(accident_xlsx_path),
        "batch_output_path": str(output_path),
        "batch_output_paths": [str(path) for path in output_paths],
        "batch_errors_path": str(error_path) if error_path.is_file() else None,
        "n_batch_records_parsed": len(parsed),
        "prompt_cache_key": cfg.effective_prompt_cache_key,
        "pass_mode": cfg.pass_mode,
        "total_tokens": total_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_cached_tokens": total_cached_tokens,
        "prompt_cache_hit_rate": prompt_cache_hit_rate,
        "sampling": sampling_stats(df_work),
        **ingest_stats,
    }
    mark_chunks_ingested(cfg)
    return final_df, meta
