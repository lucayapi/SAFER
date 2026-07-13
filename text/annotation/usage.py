"""Extraction des statistiques d'usage OpenAI (prompt caching)."""

from __future__ import annotations

from typing import Any, Dict, Optional


def default_prompt_cache_key(prompt_version: str) -> str:
    """Clé stable pour router les requêtes vers le même cache de préfixe."""
    version = str(prompt_version).strip() or "default"
    return f"safer-annotation:{version}"


def apply_prompt_cache_key(api_kwargs: Dict[str, Any], prompt_cache_key: Optional[str]) -> None:
    """
    Injecte ``prompt_cache_key`` via ``extra_body`` (appels SDK synchrone).

    Nécessaire avec ``openai<1.98`` qui ne déclare pas encore ce paramètre.
    """
    if not prompt_cache_key:
        return
    extra_body = dict(api_kwargs.get("extra_body") or {})
    extra_body["prompt_cache_key"] = str(prompt_cache_key)
    api_kwargs["extra_body"] = extra_body


def apply_prompt_cache_key_to_batch_body(
    body: Dict[str, Any],
    prompt_cache_key: Optional[str],
) -> None:
    """
    Injecte ``prompt_cache_key`` au niveau racine du corps Batch.

    L'API Batch rejette ``extra_body`` (paramètre SDK uniquement).
    """
    body.pop("extra_body", None)
    if prompt_cache_key:
        body["prompt_cache_key"] = str(prompt_cache_key)
    else:
        body.pop("prompt_cache_key", None)


def extract_usage_from_response(resp: Any) -> Dict[str, Optional[int]]:
    """
    Extrait prompt / completion / cached tokens depuis une réponse chat.completions.

    ``cached_tokens`` provient de ``usage.prompt_tokens_details`` (prompt caching OpenAI).
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {
            "usage_tokens": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "cached_tokens": None,
        }

    prompt_tokens = _as_int(getattr(usage, "prompt_tokens", None))
    completion_tokens = _as_int(getattr(usage, "completion_tokens", None))
    total_tokens = _as_int(getattr(usage, "total_tokens", None))
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    cached_tokens = _extract_cached_tokens(usage)

    return {
        "usage_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
    }


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_cached_tokens(usage: Any) -> Optional[int]:
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", None)
        if cached is not None:
            return _as_int(cached)
        if isinstance(details, dict):
            return _as_int(details.get("cached_tokens"))

    if isinstance(usage, dict):
        details_dict = usage.get("prompt_tokens_details")
        if isinstance(details_dict, dict):
            return _as_int(details_dict.get("cached_tokens"))
    return None
