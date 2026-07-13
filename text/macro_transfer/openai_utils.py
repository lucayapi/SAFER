"""Détection erreurs OpenAI (quota, rate limit) + helpers appels chat."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

ReasoningEffort = Literal["minimal", "low", "medium", "high"]

REASONING_MIN_COMPLETION_TOKENS = 2000


def _normalized_model_name(model: str) -> str:
    return str(model).strip().lower()


def openai_chat_is_reasoning_model(model: str) -> bool:
    """True pour gpt-5* et o-series (raisonnement interne avant la réponse)."""
    m = _normalized_model_name(model)
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


def openai_chat_accepts_custom_temperature(model: str) -> bool:
    """
    True si le modèle accepte une température autre que la valeur par défaut.

    Les modèles gpt-5* / o-series renvoient 400 si ``temperature != 1``.
    """
    m = _normalized_model_name(model)
    if m.startswith(("gpt-5", "o1", "o3", "o4")):
        return False
    return True


def openai_chat_uses_max_completion_tokens(model: str) -> bool:
    """True si le modèle attend ``max_completion_tokens`` au lieu de ``max_tokens``."""
    return openai_chat_is_reasoning_model(model)


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


def apply_openai_chat_max_output_tokens(
    kwargs: Dict[str, Any],
    *,
    model: str,
    max_output_tokens: Optional[int],
) -> Dict[str, Any]:
    """Ajoute la limite de sortie avec le bon nom de paramètre selon le modèle."""
    if max_output_tokens is None:
        return kwargs
    n = int(max_output_tokens)
    if openai_chat_uses_max_completion_tokens(model):
        n = max(n, REASONING_MIN_COMPLETION_TOKENS)
        kwargs["max_completion_tokens"] = n
        kwargs.pop("max_tokens", None)
    else:
        kwargs["max_tokens"] = n
        kwargs.pop("max_completion_tokens", None)
    return kwargs


def apply_openai_chat_reasoning_effort(
    kwargs: Dict[str, Any],
    *,
    model: str,
    reasoning_effort: ReasoningEffort = "minimal",
) -> Dict[str, Any]:
    """Réduit le raisonnement interne des modèles gpt-5* / o-series."""
    if openai_chat_is_reasoning_model(model):
        kwargs["reasoning_effort"] = reasoning_effort
    return kwargs


def apply_openai_chat_json_response_format(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Force une sortie JSON objet (utile pour l'annotation structurée)."""
    kwargs["response_format"] = {"type": "json_object"}
    return kwargs


def extract_chat_message_content(resp: Any) -> str:
    """Extrait le texte utile depuis une réponse chat.completions."""
    choice = resp.choices[0]
    message = choice.message
    content = getattr(message, "content", None)
    if content is not None and str(content).strip():
        return str(content).strip()

    refusal = getattr(message, "refusal", None)
    if refusal:
        return str(refusal).strip()
    return ""


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
