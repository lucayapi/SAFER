"""Vérifie l'absence de ReLU sur logit3 (macro CE) dans SCGMHead."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from scgm_text.scgm_head import SCGMHead

SCGM_HEAD_PATH = TEXT_ROOT / "scgm_text" / "scgm_head.py"
NORM_TYPES = ("logit", "weight", "logit_and_weight", "none")


def test_scgm_head_source_has_no_relu_on_logit3():
    text = SCGM_HEAD_PATH.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if "logit3" in stripped and "=" in stripped:
            assert "relu" not in stripped.lower(), f"ReLU encore présent : {stripped}"
    assert "_macro_class_logits" in text


def _batch(n_class: int = 4, n_sub: int = 8, hiddim: int = 16, batch: int = 8):
    head = SCGMHead(hiddim, n_class, n_sub)
    logit = torch.randn(batch, hiddim) - 0.5
    q = torch.zeros(batch, n_sub)
    q.scatter_(1, torch.randint(0, n_sub, (batch, 1)), 1.0)
    y = torch.zeros(batch, n_class)
    y.scatter_(1, torch.randint(0, n_class, (batch, 1)), 1.0)
    return head, logit, q, y


@pytest.mark.parametrize("norm_type", NORM_TYPES)
def test_macro_logits_shape_and_finite_loss(norm_type: str):
    head, logit, q, y = _batch()
    ls, _, _, ls3, _, _, _ = head.loss(
        logit, q, y, tau=0.1, alpha=0.5, norm_type=norm_type
    )
    assert ls3.shape == ()
    assert torch.isfinite(ls).item()
    assert torch.isfinite(ls3).item()

    logit3 = head._macro_class_logits(
        logit,
        torch.nn.functional.normalize(logit, p=2, dim=1),
        torch.nn.functional.normalize(head.mu_y, p=2, dim=1),
        norm_type,
    )
    assert logit3.shape == (8, 4)
    targets = y.argmax(1)
    assert targets.shape == (8,)


def test_macro_logits_allow_negative_dot_products():
    head, logit, q, y = _batch()
    logit = torch.full((8, 16), -1.0)
    logit3 = head._macro_class_logits(
        logit,
        torch.nn.functional.normalize(logit, p=2, dim=1),
        torch.nn.functional.normalize(head.mu_y, p=2, dim=1),
        "logit",
    )
    assert (logit3 < 0).any()


def test_loss_backward_nonzero_gradients():
    head, logit, q, y = _batch()
    logit = logit.requires_grad_(True)
    ls, _, _, _, _, _, _ = head.loss(logit, q, y, tau=0.1, alpha=0.5)
    ls.backward()
    assert head.mu_y.grad is not None
    assert head.mu_z.grad is not None
    assert float(head.mu_y.grad.abs().sum()) > 0.0


def test_macro_no_relu_log_printed_once(capsys):
    head, logit, q, y = _batch()
    head._logged_macro_no_relu = False
    head.loss(logit, q, y, tau=0.1, alpha=0.5)
    head.loss(logit, q, y, tau=0.1, alpha=0.5)
    out = capsys.readouterr().out
    assert out.count("[SCGM] macro logits computed without ReLU") == 1


def test_forward_to_logits_shape():
    head, logit, _, y = _batch()
    logit1, logit2, logit3 = head.forward_to_logits(logit, y, tau=0.1, norm_type="logit")
    assert logit1.shape == (8, 8)
    assert logit2.shape == (8, 8)
    assert logit3.shape == (8, 4)
