"""Tests sélection checkpoint SCGM sur delta_macro_pct."""

from __future__ import annotations

import sys
from pathlib import Path

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from scripts.train_scgm_text import checkpoint_selection_score


def test_checkpoint_delta_macro_pct():
    val = {"val_delta_macro_pct": 42.5, "val_eta2_macro_balanced": 0.3}
    assert checkpoint_selection_score(val, "delta_macro_pct", 0.01) == 42.5


def test_checkpoint_delta_from_eta2_fallback():
    val = {"val_eta2_macro_balanced": 0.25}
    score = checkpoint_selection_score(val, "delta_macro_pct", 0.01)
    assert abs(score - 25.0) < 1e-6
