"""Tests adaptateur TPN (CPU)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from macro_transfer.tpn_adapter import ResidualMLPAdapter, adapt_embeddings_tpn, build_adapter


def test_adapter_output_shape_and_normalized():
    dim = 16
    n = 5
    h = np.random.randn(n, dim).astype(np.float32)
    h = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
    adapter = build_adapter({"type": "residual_mlp", "init_last_zero": True}, dim)
    out = adapt_embeddings_tpn(adapter, h, device="cpu")
    assert out.shape == h.shape
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, np.ones(n), atol=1e-5)


def test_init_last_zero_near_identity():
    dim = 32
    h = np.random.randn(4, dim).astype(np.float32)
    h = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
    adapter = ResidualMLPAdapter(dim, init_last_zero=True, scale=0.1)
    adapter.eval()
    with torch.no_grad():
        t = torch.as_tensor(h)
        out = adapter(t).numpy()
    diff = np.linalg.norm(out - h, axis=1).mean()
    assert diff < 0.05
