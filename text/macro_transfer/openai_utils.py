"""Détection erreurs OpenAI (quota, rate limit)."""

from __future__ import annotations


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
