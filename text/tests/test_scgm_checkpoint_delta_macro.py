"""Tests sélection checkpoint SCGM sur train_loss."""

from __future__ import annotations

import sys
from pathlib import Path

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from scripts.train_scgm_text import checkpoint_selection_score


def test_checkpoint_train_loss_minimize():
    low = {"train_loss": 0.5}
    high = {"train_loss": 2.0}
    assert checkpoint_selection_score(low, "train_loss") > checkpoint_selection_score(high, "train_loss")
