"""Helpers for OpenAI theme labelling in results notebooks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

from manuscript_reporting import coalesce_text, sanitize_label_text


def _ensure_text_package_on_path() -> None:
    text_root = Path(__file__).resolve().parent.parent
    if str(text_root) not in sys.path:
        sys.path.insert(0, str(text_root))


def build_theme_label_chat_kwargs(
    *,
    model: str,
    messages: list[dict[str, str]],
    reasoning_effort: str = "low",
    max_output_tokens: int = 4000,
) -> dict[str, Any]:
    """Build Chat Completions kwargs aligned with the annotation pipeline (gpt-5*)."""
    _ensure_text_package_on_path()
    from macro_transfer.openai_utils import (
        apply_openai_chat_json_response_format,
        apply_openai_chat_max_output_tokens,
        apply_openai_chat_reasoning_effort,
    )

    api_kwargs: dict[str, Any] = {"model": model, "messages": messages}
    apply_openai_chat_max_output_tokens(
        api_kwargs,
        model=model,
        max_output_tokens=max_output_tokens,
    )
    apply_openai_chat_reasoning_effort(
        api_kwargs,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    apply_openai_chat_json_response_format(api_kwargs)
    return api_kwargs


def complete_theme_label_json(
    client,
    *,
    model: str,
    messages: list[dict[str, str]],
    reasoning_effort: str = "low",
    max_output_tokens: int = 4000,
) -> str:
    """Call Chat Completions and return non-empty JSON text."""
    _ensure_text_package_on_path()
    from macro_transfer.openai_utils import extract_chat_message_content

    response = client.chat.completions.create(
        **build_theme_label_chat_kwargs(
            model=model,
            messages=messages,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
        )
    )
    content = extract_chat_message_content(response)
    if not content:
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        usage = getattr(response, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        raise ValueError(
            f"Réponse vide du modèle (finish_reason={finish_reason!r}, "
            f"completion_tokens={completion_tokens}). "
            f"Augmentez OPENAI_MAX_OUTPUT_TOKENS (>= 2000 pour gpt-5*)."
        )
    return content


def parse_llm_payload(raw_text: str) -> list[dict[str, Any]]:
    """Parse the JSON object returned by the chat completion."""
    payload = json.loads(raw_text or "{}")
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("themes", payload.get("labels", payload.get("topics", [])))
        if isinstance(items, dict):
            items = [
                {"topic_id": key, **value} if isinstance(value, dict) else {"topic_id": key, "label": value}
                for key, value in items.items()
            ]
        if not items and any(key in payload for key in ("label", "llm_label", "theme", "title", "intitule")):
            items = [payload]
    else:
        items = []
    return items if isinstance(items, list) else []


def _normalize_topic_id(value) -> str:
    text = sanitize_label_text(value)
    if not text:
        return ""
    if "_" in text:
        role, suffix = text.split("_", 1)
        if suffix.isdigit():
            return f"{role}_{int(suffix):03d}"
    return text


def extract_theme_item(items: list[dict[str, Any]], record: Mapping[str, Any]) -> dict[str, Any]:
    """Match one LLM theme row to the requested topic record.

    OpenAI often omits ``topic_id`` when only one theme is requested; in that
    case accept the single returned item.
    """
    requested_id = _normalize_topic_id(record.get("topic_id"))
    normalized_items: list[dict[str, Any]] = []
    by_topic: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized_items.append(item)
        topic_id = _normalize_topic_id(item.get("topic_id"))
        if topic_id:
            by_topic[topic_id] = item
    if requested_id and requested_id in by_topic:
        return by_topic[requested_id]
    if len(normalized_items) == 1:
        return normalized_items[0]
    for item in normalized_items:
        if _normalize_topic_id(item.get("topic_id")) == requested_id:
            return item
    return {}


def normalize_llm_fields(item: Mapping[str, Any]) -> dict[str, str]:
    """Normalize label/description/evidence fields from one LLM or cache item."""
    evidence = item.get("evidence", item.get("llm_evidence", ""))
    if isinstance(evidence, list):
        evidence_text = ", ".join(sanitize_label_text(part) for part in evidence if sanitize_label_text(part))
    else:
        evidence_text = sanitize_label_text(evidence)
    return {
        "llm_label": coalesce_text(
            item.get("label"),
            item.get("llm_label"),
            item.get("intitule"),
            item.get("nom"),
            item.get("name"),
            item.get("theme"),
            item.get("title"),
            item.get("theme_name"),
        ),
        "llm_description": coalesce_text(
            item.get("description"),
            item.get("llm_description"),
            item.get("summary"),
        ),
        "llm_evidence": evidence_text,
    }


def is_valid_llm_cache_row(row: Mapping[str, Any]) -> bool:
    """True when a cached row contains a usable label."""
    return bool(sanitize_label_text(row.get("llm_label", "")))


__all__ = [
    "build_theme_label_chat_kwargs",
    "complete_theme_label_json",
    "extract_theme_item",
    "is_valid_llm_cache_row",
    "normalize_llm_fields",
    "parse_llm_payload",
]
