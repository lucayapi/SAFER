"""Ingestion des résultats JSONL Batch OpenAI (Chat Completions)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

from macro_transfer.openai_utils import extract_chat_message_content

from annotation.prompts import empty_prediction_fields
from annotation.usage import extract_usage_from_response
from annotation.validation import extract_json_object, validate_prediction


def _parse_custom_id(custom_id: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not custom_id:
        return None, None
    if "||" in custom_id:
        accident_id, fact_id = custom_id.split("||", maxsplit=1)
        return accident_id, fact_id
    return custom_id, None


def _response_body_to_content(body: Any) -> str:
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices") or []
    if not choices:
        return ""
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if content is not None and str(content).strip():
        return str(content).strip()
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, refusal=None))])
    return extract_chat_message_content(resp)


def _error_result(
    base: Dict[str, Any],
    *,
    prompt_version: str,
    pred_error: str,
    pred_raw: Optional[str] = None,
    usage: Optional[Dict[str, Any]] = None,
    http_status_code: Optional[int] = None,
) -> Dict[str, Any]:
    out = {
        **base,
        **empty_prediction_fields(prompt_version),
        "pred_raw": pred_raw,
        "pred_error": pred_error,
        "usage_tokens": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "cached_tokens": None,
        "http_status_code": http_status_code,
    }
    if usage:
        out.update(usage)
    return out


def parse_batch_result_record(
    record: Dict[str, Any],
    *,
    prompt_version: str = "v10_macro_labels_independent_outcomes",
    pass_mode: str = "pass1",
) -> Dict[str, Any]:
    """Transforme une ligne ``batch_output.jsonl`` en résultat annotation."""
    custom_id = record.get("custom_id")
    accident_id, fact_id = _parse_custom_id(custom_id)
    base = {
        "custom_id": custom_id,
        "cache_key": custom_id,
        "accident_id": accident_id,
        "fact_id": fact_id,
        "pred_source": "batch",
    }

    error = record.get("error")
    response = record.get("response")
    if error is not None or response is None:
        return _error_result(
            base,
            prompt_version=prompt_version,
            pred_error=json.dumps(error, ensure_ascii=False) if error is not None else "response absent",
        )

    status_code = int(response.get("status_code") or 0)
    body = response.get("body") or {}
    content = _response_body_to_content(body)
    usage = extract_usage_from_response(SimpleNamespace(usage=body.get("usage")))

    if status_code != 200 or not content:
        return _error_result(
            base,
            prompt_version=prompt_version,
            pred_raw=content or None,
            pred_error=f"HTTP {status_code} ou contenu vide",
            usage=usage,
            http_status_code=status_code,
        )

    try:
        validated = validate_prediction(
            extract_json_object(content),
            prompt_version=prompt_version,
            pass_mode=pass_mode,
        )
        return {
            **base,
            **validated,
            "pred_raw": content,
            "pred_error": None,
            **usage,
            "http_status_code": status_code,
        }
    except ValueError as exc:
        preview = content[:500].replace("\n", " ")
        return _error_result(
            base,
            prompt_version=prompt_version,
            pred_raw=content,
            pred_error=f"{exc} | raw_preview={preview!r}",
            usage=usage,
            http_status_code=status_code,
        )


def iter_batch_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                yield {
                    "custom_id": None,
                    "error": {"message": f"Ligne {line_number}: JSON invalide ({exc})"},
                    "response": None,
                }


def parse_batch_output_jsonl(
    path: Path,
    *,
    extra_paths: Optional[List[Path]] = None,
    prompt_version: str = "v10_macro_labels_independent_outcomes",
    pass_mode: str = "pass1",
) -> List[Dict[str, Any]]:
    paths = [path]
    if extra_paths:
        paths.extend(extra_paths)
    results: List[Dict[str, Any]] = []
    for jsonl_path in paths:
        if not jsonl_path.is_file():
            continue
        for record in iter_batch_jsonl(jsonl_path):
            results.append(
                parse_batch_result_record(
                    record,
                    prompt_version=prompt_version,
                    pass_mode=pass_mode,
                )
            )
    return results


def results_by_cache_key(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for res in results:
        key = res.get("cache_key") or res.get("custom_id")
        if key:
            mapping[str(key)] = res
    return mapping
