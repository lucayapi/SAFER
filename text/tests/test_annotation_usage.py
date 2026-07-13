"""Tests extraction usage OpenAI (prompt caching)."""

from __future__ import annotations

from types import SimpleNamespace

from annotation.usage import (
    apply_prompt_cache_key,
    apply_prompt_cache_key_to_batch_body,
    default_prompt_cache_key,
    extract_usage_from_response,
)


def test_default_prompt_cache_key():
    assert default_prompt_cache_key("v10_macro_labels_independent_outcomes") == (
        "safer-annotation:v10_macro_labels_independent_outcomes"
    )


def test_apply_prompt_cache_key_uses_extra_body_on_old_sdk():
    api_kwargs: dict = {"model": "gpt-5-nano", "messages": []}
    apply_prompt_cache_key(api_kwargs, "safer-annotation:v10")
    assert "prompt_cache_key" not in api_kwargs
    assert api_kwargs["extra_body"]["prompt_cache_key"] == "safer-annotation:v10"


def test_apply_prompt_cache_key_to_batch_body_top_level():
    body: dict = {"model": "gpt-5-mini", "extra_body": {"prompt_cache_key": "old"}}
    apply_prompt_cache_key_to_batch_body(body, "safer-annotation:v13")
    assert "extra_body" not in body
    assert body["prompt_cache_key"] == "safer-annotation:v13"


def test_extract_usage_from_response_with_cached_tokens():
    resp = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=2800,
            completion_tokens=120,
            total_tokens=2920,
            prompt_tokens_details=SimpleNamespace(cached_tokens=2300),
        )
    )
    usage = extract_usage_from_response(resp)
    assert usage["usage_tokens"] == 2920
    assert usage["prompt_tokens"] == 2800
    assert usage["completion_tokens"] == 120
    assert usage["cached_tokens"] == 2300


def test_extract_usage_without_cached_tokens():
    resp = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=500,
            completion_tokens=50,
            total_tokens=550,
            prompt_tokens_details=None,
        )
    )
    usage = extract_usage_from_response(resp)
    assert usage["cached_tokens"] is None
    assert usage["usage_tokens"] == 550
