"""Préparation du fichier JSONL d'entrée pour l'API Batch OpenAI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from annotation.batch_config import BatchAnnotationConfig
from annotation.batch_request import build_batch_request_line
from annotation.cache import get_batch_paths, get_output_paths, load_cache, make_cache_key, save_run_config, should_skip_row_for_batch
from annotation.runner import prepare_annotation_frame


def load_input_dataframe(cfg: BatchAnnotationConfig) -> pd.DataFrame:
    path = cfg.resolved_input_path
    if not path.is_file():
        raise FileNotFoundError(f"CSV introuvable : {path}")
    return pd.read_csv(path)


def filter_rows_for_batch(
    df: pd.DataFrame,
    cfg: BatchAnnotationConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Exclut les lignes déjà annotées avec succès dans le cache JSONL."""
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path, _, _, _, _ = get_output_paths(
        cfg.outputs_dir,
        model_id=cfg.openai_model,
        prompt_version=cfg.prompt_version,
        artifact_slug=cfg.artifact_slug,
    )
    cache = {} if cfg.skip_cache else load_cache(jsonl_path)

    keep_indices: list[int] = []
    cache_hits = 0
    batch_exhausted = 0
    for idx, row in df.iterrows():
        cache_key = make_cache_key(row, row_idx=idx)
        cached = cache.get(cache_key)
        if cached and cached.get("pred_ok"):
            cache_hits += 1
            continue
        if should_skip_row_for_batch(cached, max_batch_retries=cfg.max_batch_retries):
            batch_exhausted += 1
            continue
        keep_indices.append(int(idx))

    filtered = df.loc[keep_indices].copy()
    filtered["_source_row_idx"] = keep_indices
    filtered = filtered.reset_index(drop=True)
    stats = {
        "n_input_rows": int(len(df)),
        "n_cache_hits": cache_hits,
        "n_batch_exhausted": batch_exhausted,
        "n_batch_requests": int(len(filtered)),
        "jsonl_cache_path": str(jsonl_path),
    }
    return filtered, stats


def write_batch_input_jsonl(
    df: pd.DataFrame,
    cfg: BatchAnnotationConfig,
    *,
    output_path: Path,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(output_path, "w", encoding="utf-8") as handle:
        for idx, row in df.iterrows():
            source_idx = row.get("_source_row_idx", idx)
            row_idx = int(source_idx) if pd.notna(source_idx) else int(idx)
            cache_key = make_cache_key(row, row_idx=row_idx)
            line = build_batch_request_line(
                custom_id=cache_key,
                row=row,
                cfg=cfg,
                endpoint=cfg.batch_endpoint,
            )
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
            n_written += 1
    return {
        "batch_input_path": str(output_path),
        "n_requests_written": n_written,
        "file_size_bytes": int(output_path.stat().st_size),
    }


def write_batch_input_jsonl_chunks(
    df: pd.DataFrame,
    cfg: BatchAnnotationConfig,
    *,
    output_dir: Path,
    base_stem: str,
) -> dict[str, Any]:
    """Découpe le JSONL d'entrée (limite OpenAI Batch : 200 Mo / 50k requêtes)."""
    if df.empty:
        return {
            "batch_input_chunks": [],
            "batch_input_path": None,
            "n_requests_written": 0,
            "n_chunks": 0,
        }

    max_per = max(1, int(cfg.max_requests_per_batch))
    chunks: list[dict[str, Any]] = []
    total_written = 0

    for start in range(0, len(df), max_per):
        chunk_index = len(chunks)
        chunk_df = df.iloc[start : start + max_per].reset_index(drop=True)
        if len(df) <= max_per:
            output_path = output_dir / f"{base_stem}__batch_input.jsonl"
        else:
            output_path = output_dir / f"{base_stem}__batch_input__part{chunk_index + 1:03d}.jsonl"
        stats = write_batch_input_jsonl(chunk_df, cfg, output_path=output_path)
        chunks.append(
            {
                "chunk_index": chunk_index,
                "batch_input_path": str(output_path),
                "n_requests": int(stats["n_requests_written"]),
                "file_size_bytes": int(stats["file_size_bytes"]),
            }
        )
        total_written += int(stats["n_requests_written"])

    return {
        "batch_input_chunks": chunks,
        "batch_input_path": chunks[0]["batch_input_path"] if len(chunks) == 1 else None,
        "n_requests_written": total_written,
        "n_chunks": len(chunks),
    }


def prepare_batch_input(cfg: BatchAnnotationConfig) -> dict[str, Any]:
    """Charge les données, filtre le cache, écrit le JSONL batch."""
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(cfg.outputs_dir, cfg.to_dict())

    df_raw = load_input_dataframe(cfg)
    df_work = prepare_annotation_frame(cfg, df_raw)
    df_todo, filter_stats = filter_rows_for_batch(df_work, cfg)

    batch_input_path, _, _, _ = get_batch_paths(
        cfg.outputs_dir,
        model_id=cfg.openai_model,
        prompt_version=cfg.prompt_version,
        artifact_slug=cfg.artifact_slug,
    )
    base_stem = batch_input_path.name.replace("__batch_input.jsonl", "")
    write_stats = write_batch_input_jsonl_chunks(
        df_todo,
        cfg,
        output_dir=cfg.outputs_dir,
        base_stem=base_stem,
    )

    return {
        "run_id": cfg.run_id,
        "outputs_dir": str(cfg.outputs_dir),
        "dataframe": df_work,
        "todo_dataframe": df_todo,
        **filter_stats,
        **write_stats,
    }
