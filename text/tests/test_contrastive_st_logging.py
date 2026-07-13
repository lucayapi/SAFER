"""Tests log train_log.csv unifié (entraînement contrastif HF)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.training_log import (
    TRAIN_LOG_COLUMNS,
    build_train_log_row,
    mean_train_loss_for_epoch,
)


def test_mean_train_loss_for_epoch_averages_rows():
    history = [
        {"epoch": 1, "train_loss": 0.4},
        {"epoch": 1, "train_loss": 0.2},
        {"epoch": 2, "train_loss": 0.1},
    ]
    assert mean_train_loss_for_epoch(history, 1) == pytest.approx(0.3)
    assert mean_train_loss_for_epoch(history, 2) == pytest.approx(0.1)


def test_build_train_log_row_val_loss():
    row = build_train_log_row(1, 0.5, val_loss=0.4)
    assert row["epoch"] == 1
    assert row["train_loss"] == 0.5
    assert row["val_loss"] == 0.4


def test_train_log_columns_list():
    assert TRAIN_LOG_COLUMNS == ["epoch", "train_loss", "val_loss"]
