"""Détection erreurs OpenAI (quota, rate limit) + helpers appels chat."""

from __future__ import annotations

from typing import Any, Dict, Optional


def openai_chat_accepts_custom_temperature(model: str) -> bool:
    """
    True si le modèle accepte une température autre que la valeur par défaut.

    Les modèles gpt-5* / o-series renvoient 400 si ``temperature != 1``.
    """
    m = str(model).strip().lower()
    if m.startswith(("gpt-5", "o1", "o3", "o4")):
        return False
    return True


def apply_openai_chat_temperature(
    kwargs: Dict[str, Any],
    *,
    model: str,
    temperature: Optional[float],
) -> Dict[str, Any]:
    """Ajoute ``temperature`` à ``kwargs`` seulement si le modèle le supporte."""
    if temperature is not None and openai_chat_accepts_custom_temperature(model):
        kwargs["temperature"] = float(temperature)
    return kwargs


def is_openai_capacity_error(exc: BaseException) -> bool:
    """True si l'erreur ressemble à un quota dépassé ou un rate limit (429)."""
    try:
        from openai import APIStatusError, RateLimitError

        if isinstance(exc, RateLimitError):
            return True
        if isinstance(exc, APIStatusError):
            code = getattr(exc, "status_code", None)
            if code == 429:
                return True
            body = getattr(exc, "body", None)
            if isinstance(body, dict):
                err = body.get("error") or {}
                if str(err.get("code", "")).lower() in {"insufficient_quota", "rate_limit_exceeded"}:
                    return True
    except ImportError:
        pass

    name = type(exc).__name__
    if name in {"RateLimitError", "APIStatusError"}:
        return True

    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "429",
            "rate limit",
            "rate_limit",
            "insufficient_quota",
            "exceeded your current quota",
            "too many requests",
        )
    )
