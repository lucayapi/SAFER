"""Tests build_topic_embeddings (espace BERTopic TPN)."""

from __future__ import annotations

import numpy as np
import pytest

from macro_transfer.topic_embeddings import (
    build_topic_embeddings,
    resolve_topic_embedding_cfg,
)


def _rand(n: int = 8, d: int = 16) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, d))
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def test_initial_mode():
    h = _rand()
    out = build_topic_embeddings(h, mode="initial")
    np.testing.assert_allclose(out, h, rtol=1e-5)


def test_adapted_mode():
    h0 = _rand()
    h1 = _rand()
    out = build_topic_embeddings(h0, h1, mode="adapted")
    np.testing.assert_allclose(out, h1, rtol=1e-5)


def test_mixed_mode():
    h0 = _rand()
    h1 = _rand()
    out = build_topic_embeddings(h0, h1, mode="mixed", alpha=0.5, normalize=False)
    expected = 0.5 * h0 + 0.5 * h1
    np.testing.assert_allclose(out, expected, rtol=1e-5)


def test_l2_normalize_output():
    h0 = _rand()
    h1 = _rand()
    out = build_topic_embeddings(h0, h1, mode="mixed", alpha=0.25, normalize=True)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, np.ones(len(out)), rtol=1e-5, atol=1e-5)


def test_adapted_missing_raises():
    h = _rand()
    with pytest.raises(ValueError, match="h_adapted requis"):
        build_topic_embeddings(h, mode="adapted")
    with pytest.raises(ValueError, match="h_adapted requis"):
        build_topic_embeddings(h, mode="mixed")


def test_resolve_topic_embedding_cfg_cli_overrides_yaml():
    cfg = {"embedding_space": {"mode": "initial", "alpha": 0.0, "normalize": True}}
    resolved = resolve_topic_embedding_cfg(cfg, cli_mode="mixed", cli_alpha=0.75)
    assert resolved["mode"] == "mixed"
    assert resolved["alpha"] == 0.75
