"""Tests selection_score géométrie."""

from __future__ import annotations

import sys
from pathlib import Path

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.eval_geometry import selection_score


def test_selection_score_prefers_higher_delta():
    low = {"eta2_macro_balanced_perc": 10.0}
    high = {"eta2_macro_balanced_perc": 40.0}
    assert selection_score(high) > selection_score(low)
