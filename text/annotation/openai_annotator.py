"""Appels API OpenAI pour annotation d'unités factuelles."""

from __future__ import annotations

import time
import traceback
from typing import Any, Dict, Mapping, Optional

from macro_transfer.openai_utils import extract_chat_message_content, is_openai_capacity_error
from scgm_text.openai_theme_labels import _get_client, load_openai_dotenv

from annotation.batch_request import build_chat_completion_body
from annotation.config import AnnotationConfig
from annotation.prompts import empty_prediction_fields
from annotation.usage import extract_usage_from_response
from annotation.validation import extract_json_object, validate_prediction


def call_openai_annotation(
    row: Mapping[str, Any],
    *,
    cfg: AnnotationConfig,
    first_pass_annotation: Optional[Mapping[str, Any]] = None,
    client: Any = None,
) -> Dict[str, Any]:
    load_openai_dotenv()
    if client is None:
        client = _get_client()

    last_error: Optional[str] = None
    for attempt in range(cfg.max_retries + 1):
        try:
            if cfg.min_delay_between_calls_sec > 0:
                time.sleep(float(cfg.min_delay_between_calls_sec))

            api_kwargs = build_chat_completion_body(
                row,
                cfg,
                first_pass_annotation=first_pass_annotation,
            )
            resp = client.chat.completions.create(**api_kwargs)
            content = extract_chat_message_content(resp)
            if not content:
                finish_reason = getattr(resp.choices[0], "finish_reason", None)
                usage = extract_usage_from_response(resp)
                raise ValueError(
                    "Réponse vide du modèle "
                    f"(finish_reason={finish_reason!r}, "
                    f"completion_tokens={usage.get('completion_tokens')}). "
                    "Les modèles gpt-5 consomment des tokens en raisonnement interne : "
                    "augmentez max_output_tokens (>= 4000 pour reasoning_effort=medium)."
                )
            try:
                validated = validate_prediction(
                    extract_json_object(content),
                    prompt_version=cfg.prompt_version,
                    pass_mode=cfg.pass_mode,
                )
            except ValueError as exc:
                preview = content[:500].replace("\n", " ")
                raise ValueError(f"{exc} | raw_preview={preview!r}") from exc
            usage = extract_usage_from_response(resp)
            return {
                **validated,
                "pred_raw": content,
                "pred_error": None,
                **usage,
            }
        except Exception as exc:
            sleep_s = float(cfg.retry_base_sleep_sec) * (2**attempt)
            if is_openai_capacity_error(exc):
                sleep_s = max(sleep_s, float(cfg.rate_limit_sleep_sec))
            last_error = (
                f"Tentative {attempt + 1}/{cfg.max_retries + 1} échouée : "
                f"{type(exc).__name__}: {repr(exc)}\n{traceback.format_exc()}"
            )
            time.sleep(sleep_s)

    return {
        **empty_prediction_fields(cfg.prompt_version),
        "pred_raw": None,
        "pred_error": last_error,
        "usage_tokens": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "cached_tokens": None,
    }
