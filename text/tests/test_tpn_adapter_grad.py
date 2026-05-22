"""Tests gradient TPN (_compute_tpn_losses entièrement Torch)."""

from __future__ import annotations

import torch

from macro_transfer.constants import MACRO_NAMES
from macro_transfer.tpn_adapter import ResidualMLPAdapter, _compute_tpn_losses


def _synthetic_batch(*, dim: int = 16, n_s: int = 8, n_t: int = 8):
    torch.manual_seed(0)
    h_s = torch.randn(n_s, dim)
    h_t = torch.randn(n_t, dim)
    h_s = torch.nn.functional.normalize(h_s, p=2, dim=-1)
    h_t = torch.nn.functional.normalize(h_t, p=2, dim=-1)
    y_ids = torch.tensor([i % len(MACRO_NAMES) for i in range(n_s)], dtype=torch.long)
    return h_s, h_t, y_ids


def _adapter(dim: int = 16) -> ResidualMLPAdapter:
    return ResidualMLPAdapter(dim, bottleneck_dim=32, scale=0.1)


def _zero_weights():
    return {
        "src": 0.0,
        "proto": 0.0,
        "kl": 0.0,
        "ent": 0.0,
        "div": 0.0,
        "preserve": 0.0,
    }


def test_loss_total_requires_grad_and_backward():
    h_s, h_t, y_ids = _synthetic_batch()
    adapter = _adapter()
    adapter.train()
    losses = _compute_tpn_losses(
        adapter,
        h_s,
        h_t,
        y_ids,
        tpn_cfg={"tau": 0.3, "distance_metric": "euclidean", "detach_assignments": True},
        loss_weights={"src": 1.0, "proto": 1.0, "kl": 0.5, "ent": 0.05, "div": 0.05, "preserve": 0.1},
    )
    assert isinstance(losses["loss_total"], torch.Tensor)
    assert losses["loss_total"].requires_grad
    adapter.zero_grad()
    losses["loss_total"].backward()
    grads = [p.grad for p in adapter.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_proto_only_grad():
    h_s, h_t, y_ids = _synthetic_batch()
    adapter = _adapter()
    adapter.train()
    w = _zero_weights()
    w["proto"] = 1.0
    losses = _compute_tpn_losses(
        adapter,
        h_s,
        h_t,
        y_ids,
        tpn_cfg={"tau": 0.3, "distance_metric": "euclidean"},
        loss_weights=w,
    )
    adapter.zero_grad()
    losses["loss_total"].backward()
    grad_norm = sum(p.grad.norm().item() ** 2 for p in adapter.parameters() if p.grad is not None) ** 0.5
    assert grad_norm > 1e-8


def test_detach_assignments_false_no_crash():
    h_s, h_t, y_ids = _synthetic_batch()
    adapter = _adapter()
    adapter.train()
    losses = _compute_tpn_losses(
        adapter,
        h_s,
        h_t,
        y_ids,
        tpn_cfg={
            "tau": 0.3,
            "distance_metric": "euclidean",
            "detach_assignments": False,
            "assignment_mode": "soft",
        },
        loss_weights={"src": 1.0, "proto": 1.0, "kl": 0.5, "ent": 0.05, "div": 0.05, "preserve": 0.1},
    )
    assert losses["loss_total"].requires_grad
    losses["loss_total"].backward()


def test_entropy_only_grad():
    h_s, h_t, y_ids = _synthetic_batch()
    adapter = _adapter()
    adapter.train()
    w = _zero_weights()
    w["ent"] = 1.0
    losses = _compute_tpn_losses(
        adapter,
        h_s,
        h_t,
        y_ids,
        tpn_cfg={"tau": 0.3, "distance_metric": "euclidean"},
        loss_weights=w,
    )
    assert losses["loss_total"].requires_grad
    adapter.zero_grad()
    losses["loss_total"].backward()
    grad_norm = sum(p.grad.norm().item() ** 2 for p in adapter.parameters() if p.grad is not None) ** 0.5
    assert grad_norm > 1e-8
