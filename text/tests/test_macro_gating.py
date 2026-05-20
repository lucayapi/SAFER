"""Tests gating macro."""

from __future__ import annotations

import numpy as np
import pytest

from macro_transfer.constants import MACRO_NAMES
from macro_transfer.gating import apply_macro_gating


def test_gating_argmax_and_threshold():
    p = np.array(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.4],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )
    g = apply_macro_gating(p, confidence_threshold=0.5)
    assert list(g["m_hat"]) == ["A0", "C", MACRO_NAMES[int(np.argmax(p[2]))]]
    assert g["q_conf"].iloc[0] == pytest.approx(0.7)
    assert bool(g["ambiguous"].iloc[0]) is False
    assert bool(g["ambiguous"].iloc[1]) is True
    assert bool(g["ambiguous"].iloc[2]) is True
