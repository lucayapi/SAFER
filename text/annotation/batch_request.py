"""Construction des requêtes Chat Completions pour l'API Batch OpenAI."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from macro_transfer.openai_utils import (
    apply_openai_chat_json_response_format,
    apply_openai_chat_max_output_tokens,
    apply_openai_chat_reasoning_effort,
    apply_openai_chat_temperature,
)

from annotation.config import AnnotationConfig
from annotation.prompts import resolve_prompt_bundle
from annotation.usage import apply_prompt_cache_key, apply_prompt_cache_key_to_batch_body


def build_chat_completion_body(
    row: Mapping[str, Any],
    cfg: AnnotationConfig,
    *,
    summary_col: Optional[str] = None,
    first_pass_annotation: Optional[Mapping[str, Any]] = None,
    for_batch: bool = False,
) -> Dict[str, Any]:
    """Corps ``body`` d'une requête ``/v1/chat/completions`` (sync ou batch)."""
    col = summary_col or cfg.summary_col
    pass_mode = getattr(cfg, "pass_mode", "pass1")
    bundle = resolve_prompt_bundle(
        cfg.prompt_version,
        pass_mode=pass_mode,
        summary_col=col,
    )
    build_user_prompt = bundle["build_user_prompt"]
    user_content = build_user_prompt(row, first_pass_annotation=first_pass_annotation)
    messages = [
        {"role": "system", "content": bundle["system_prompt"]},
        {"role": "user", "content": user_content},
    ]
    api_kwargs: Dict[str, Any] = {
        "model": cfg.openai_model,
        "messages": messages,
    }
    apply_openai_chat_max_output_tokens(
        api_kwargs,
        model=cfg.openai_model,
        max_output_tokens=cfg.max_output_tokens,
    )
    apply_openai_chat_reasoning_effort(
        api_kwargs,
        model=cfg.openai_model,
        reasoning_effort=cfg.reasoning_effort,
    )
    apply_openai_chat_json_response_format(api_kwargs)
    if for_batch:
        apply_prompt_cache_key_to_batch_body(api_kwargs, cfg.effective_prompt_cache_key)
    elif cfg.effective_prompt_cache_key:
        apply_prompt_cache_key(api_kwargs, cfg.effective_prompt_cache_key)
    apply_openai_chat_temperature(
        api_kwargs,
        model=cfg.openai_model,
        temperature=cfg.temperature,
    )
    return api_kwargs


def build_batch_request_line(
    *,
    custom_id: str,
    row: Mapping[str, Any],
    cfg: AnnotationConfig,
    endpoint: str = "/v1/chat/completions",
) -> Dict[str, Any]:
    """Une ligne du fichier JSONL d'entrée Batch OpenAI."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": endpoint,
        "body": build_chat_completion_body(row, cfg, for_batch=True),
    }
