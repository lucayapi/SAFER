"""Boucle d'annotation avec cache JSONL et snapshots XLSX."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm.auto import tqdm

from annotation.aggregate import aggregate_outcomes_by_accident, summarize_predictions
from annotation.cache import (
    append_to_cache,
    get_output_paths,
    load_cache,
    make_cache_key,
    save_run_config,
)
from annotation.config import AnnotationConfig
from annotation.export_io import (
    ANNOTATION_TABLE_SUFFIX,
    attach_accident_summary_column,
    reorder_annotation_output_columns,
    save_annotation_table,
)
from annotation.openai_annotator import call_openai_annotation
from annotation.prompts import empty_prediction_fields
from annotation.sampling import sample_accidents_and_units, sampling_stats
from annotation.two_pass import (
    build_first_pass_annotation,
    load_pass1_annotated,
    merge_pass1_pass2,
)
from annotation.validation import extract_context_used


def _context_used_from_cache(cached: Dict[str, Any]) -> Optional[bool]:
    value = cached.get("pred_context_used")
    if value is not None:
        return bool(value)
    justification = cached.get("pred_justification")
    if not justification:
        return None
    try:
        return extract_context_used(str(justification))
    except ValueError:
        return None


def _v13_fields_from_result(res: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pred_ambiguous": res.get("pred_ambiguous"),
        "pred_context_needed": res.get("pred_context_needed"),
        "pred_alternative_label": res.get("pred_alternative_label"),
        "pred_ambiguity_type": res.get("pred_ambiguity_type"),
        "pred_ambiguity_reason": res.get("pred_ambiguity_reason"),
    }


def _cache_record_from_result(
    *,
    cache_key: str,
    cfg: AnnotationConfig,
    row: pd.Series,
    res: Dict[str, Any],
) -> Dict[str, Any]:
    record = {
        "cache_key": cache_key,
        "model_id": cfg.openai_model,
        "prompt_version": cfg.prompt_version,
        "pass_mode": cfg.pass_mode,
        "prompt_cache_key": cfg.effective_prompt_cache_key,
        "accident_id": row.get("accident_id", ""),
        "fact_id": row.get("fact_id", ""),
        "sentence": row.get("sentence", ""),
        "pred_label": res.get("pred_label"),
        "pred_injury_mentioned": res.get("pred_injury_mentioned"),
        "pred_hospitalized": res.get("pred_hospitalized"),
        "pred_fatal": res.get("pred_fatal"),
        "pred_confidence": res.get("pred_confidence"),
        "pred_justification": res.get("pred_justification"),
        "pred_context_used": res.get("pred_context_used"),
        "pred_ok": res.get("pred_ok"),
        "pred_raw": res.get("pred_raw"),
        "pred_error": res.get("pred_error"),
        "usage_tokens": res.get("usage_tokens"),
        "prompt_tokens": res.get("prompt_tokens"),
        "completion_tokens": res.get("completion_tokens"),
        "cached_tokens": res.get("cached_tokens"),
        "saved_at": pd.Timestamp.utcnow().isoformat(),
    }
    record.update(_v13_fields_from_result(res))
    return record


def _result_from_cache(cached: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "pred_label": cached.get("pred_label"),
        "pred_injury_mentioned": cached.get("pred_injury_mentioned"),
        "pred_hospitalized": cached.get("pred_hospitalized"),
        "pred_fatal": cached.get("pred_fatal"),
        "pred_confidence": cached.get("pred_confidence"),
        "pred_justification": cached.get("pred_justification"),
        "pred_context_used": _context_used_from_cache(cached),
        "pred_ok": cached.get("pred_ok"),
        "pred_raw": cached.get("pred_raw"),
        "pred_error": cached.get("pred_error"),
        "pred_source": "cache",
        "usage_tokens": cached.get("usage_tokens"),
        "prompt_tokens": cached.get("prompt_tokens"),
        "completion_tokens": cached.get("completion_tokens"),
        "cached_tokens": cached.get("cached_tokens"),
    }
    result.update(_v13_fields_from_result(cached))
    return result


def _lock_pass1_outcomes(res: Dict[str, Any], row: pd.Series) -> Dict[str, Any]:
    """En passe 2, conserve injury/hospitalized/fatal de la passe 1."""
    out = dict(res)
    for field in ("pred_injury_mentioned", "pred_hospitalized", "pred_fatal"):
        if field in row.index and pd.notna(row[field]):
            out[field] = row[field]
    return out


def _final_annotated_path(annotated_xlsx_path: Path) -> Path:
    return annotated_xlsx_path.with_name(
        annotated_xlsx_path.name.replace(
            f"__annotated{ANNOTATION_TABLE_SUFFIX}",
            f"__annotated_final{ANNOTATION_TABLE_SUFFIX}",
        )
    )


_RESULT_MERGE_DROP_COLS = (
    "accident_id",
    "fact_id",
    "custom_id",
    "cache_key",
    "http_status_code",
)


def results_dataframe_for_merge(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """DataFrame résultats sans colonnes déjà présentes dans le CSV source."""
    frame = pd.DataFrame(results).reset_index(drop=True)
    drop = [col for col in _RESULT_MERGE_DROP_COLS if col in frame.columns]
    if drop:
        frame = frame.drop(columns=drop)
    return frame


def prepare_annotation_frame(cfg: AnnotationConfig, df: pd.DataFrame) -> pd.DataFrame:
    cfg.validate_input_columns(list(df.columns))
    sampled = sample_accidents_and_units(
        df,
        n_accidents=cfg.n_accidents,
        units_per_accident=cfg.units_per_accident,
        seed=cfg.accident_sample_seed,
        accident_sample_frac=cfg.accident_sample_frac,
    )
    return sampled.reset_index(drop=True)


def classify_dataframe_with_cache(
    df: pd.DataFrame,
    cfg: AnnotationConfig,
    *,
    show_errors: bool = True,
    client: Any = None,
    pass1_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(cfg.outputs_dir, cfg.to_dict())

    (
        jsonl_path,
        snapshot_xlsx_path,
        annotated_xlsx_path,
        summary_xlsx_path,
        accident_xlsx_path,
    ) = get_output_paths(
        cfg.outputs_dir,
        model_id=cfg.openai_model,
        prompt_version=cfg.prompt_version,
        artifact_slug=cfg.artifact_slug,
    )
    final_annotated_path = _final_annotated_path(annotated_xlsx_path)

    if pass1_df is None and cfg.pass_mode == "pass2" and cfg.pass1_run_id:
        pass1_df = load_pass1_annotated(
            cfg.pass1_run_id,
            annotation_root=Path(cfg.annotation_root),
            openai_model=cfg.openai_model,
            prompt_version=cfg.prompt_version,
            pass_mode="pass1",
        )

    cache = {} if cfg.skip_cache else load_cache(jsonl_path)
    tqdm.write(f"Cache JSONL : {len(cache)} entrées depuis {jsonl_path}")
    if cfg.effective_prompt_cache_key:
        tqdm.write(f"Prompt caching OpenAI : prompt_cache_key={cfg.effective_prompt_cache_key!r}")

    results: List[Dict[str, Any]] = []
    new_processed = 0
    cache_hits = 0
    total_tokens = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cached_tokens = 0

    progress_bar = tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Annotation OpenAI [{cfg.openai_model}] ({cfg.pass_mode})",
        unit="ligne",
    )

    for idx, row in progress_bar:
        cache_key = make_cache_key(row, row_idx=idx)
        cached_entry = None if cfg.skip_cache else cache.get(cache_key)
        if cached_entry and cached_entry.get("pred_ok"):
            res = _result_from_cache(cached_entry)
            if cfg.pass_mode == "pass2":
                res = _lock_pass1_outcomes(res, row)
            cache_hits += 1
        elif cfg.dry_run:
            res = {
                **empty_prediction_fields(cfg.prompt_version),
                "pred_error": "DRY_RUN",
                "pred_source": "dry_run",
                "pred_raw": None,
                "usage_tokens": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "cached_tokens": None,
            }
        else:
            first_pass_annotation = None
            if cfg.pass_mode == "pass2":
                first_pass_annotation = build_first_pass_annotation(row)

            res = call_openai_annotation(
                row,
                cfg=cfg,
                first_pass_annotation=first_pass_annotation,
                client=client,
            )
            if cfg.pass_mode == "pass2":
                res = _lock_pass1_outcomes(res, row)

            res["pred_source"] = "new"
            cache_record = _cache_record_from_result(
                cache_key=cache_key,
                cfg=cfg,
                row=row,
                res=res,
            )
            append_to_cache(jsonl_path, cache_record)
            cache[cache_key] = cache_record
            new_processed += 1
            if res.get("usage_tokens"):
                total_tokens += int(res["usage_tokens"])
            if res.get("prompt_tokens"):
                total_prompt_tokens += int(res["prompt_tokens"])
            if res.get("completion_tokens"):
                total_completion_tokens += int(res["completion_tokens"])
            if res.get("cached_tokens"):
                total_cached_tokens += int(res["cached_tokens"])

            if show_errors and not res.get("pred_ok"):
                sentence = str(row.get("sentence", ""))[:180].replace("\n", " ")
                tqdm.write(
                    f"[ERREUR] idx={idx} | accident_id={row.get('accident_id', '')} | "
                    f"fact_id={row.get('fact_id', '')}\nPhrase: {sentence}\n"
                    f"Détail: {res.get('pred_error')}\n"
                )

        results.append(res)
        n_done = len(results)
        n_ok = sum(bool(r.get("pred_ok")) for r in results)
        progress_bar.set_postfix(
            {
                "ok": n_ok,
                "err": n_done - n_ok,
                "jsonl": cache_hits,
                "new": new_processed,
                "cached_tok": total_cached_tokens,
            }
        )

        if n_done % max(1, int(cfg.save_every)) == 0:
            tmp_df = pd.concat(
                [df.iloc[:n_done].reset_index(drop=True), results_dataframe_for_merge(results)],
                axis=1,
            )
            tmp_df = reorder_annotation_output_columns(
                attach_accident_summary_column(tmp_df, summary_col=cfg.summary_col)
            )
            save_annotation_table(tmp_df, snapshot_xlsx_path)
            tqdm.write(f"Snapshot sauvegardé : {snapshot_xlsx_path}")

    final_df = pd.concat(
        [df.reset_index(drop=True), results_dataframe_for_merge(results)],
        axis=1,
    )
    final_df = reorder_annotation_output_columns(
        attach_accident_summary_column(final_df, summary_col=cfg.summary_col)
    )
    save_annotation_table(final_df, annotated_xlsx_path)
    tqdm.write(f"Sauvegarde finale : {annotated_xlsx_path}")

    merged_df = final_df
    if cfg.pass_mode == "pass2" and pass1_df is not None:
        merged_df = merge_pass1_pass2(pass1_df, final_df)
        merged_df = reorder_annotation_output_columns(
            attach_accident_summary_column(merged_df, summary_col=cfg.summary_col)
        )
        save_annotation_table(merged_df, final_annotated_path)
        tqdm.write(f"Fusion pass1+pass2 : {final_annotated_path}")

    summary_df = summarize_predictions(merged_df)
    save_annotation_table(summary_df, summary_xlsx_path)
    accident_df = aggregate_outcomes_by_accident(merged_df, summary_col=cfg.summary_col)
    save_annotation_table(accident_df, accident_xlsx_path)

    prompt_cache_hit_rate = None
    if total_prompt_tokens > 0 and total_cached_tokens > 0:
        prompt_cache_hit_rate = float(total_cached_tokens) / float(total_prompt_tokens)

    meta = {
        "jsonl_path": str(jsonl_path),
        "snapshot_xlsx_path": str(snapshot_xlsx_path),
        "annotated_xlsx_path": str(annotated_xlsx_path),
        "annotated_final_xlsx_path": str(final_annotated_path)
        if cfg.pass_mode == "pass2"
        else None,
        "summary_xlsx_path": str(summary_xlsx_path),
        "accident_xlsx_path": str(accident_xlsx_path),
        "prompt_cache_key": cfg.effective_prompt_cache_key,
        "pass_mode": cfg.pass_mode,
        "pass1_run_id": cfg.pass1_run_id,
        "cache_hits": cache_hits,
        "new_processed": new_processed,
        "total_tokens": total_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_cached_tokens": total_cached_tokens,
        "prompt_cache_hit_rate": prompt_cache_hit_rate,
        "sampling": sampling_stats(df),
    }
    return merged_df, meta
