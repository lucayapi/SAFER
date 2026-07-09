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
    EpochLossAccumulator,
    build_train_log_row,
    mean_train_loss_for_epoch,
)


def test_mean_train_loss_for_epoch_uses_accumulator():
    acc = EpochLossAccumulator()
    acc.record(0.4, 0.5)
    acc.record(0.2, 0.9)
    assert mean_train_loss_for_epoch([], 1, loss_accumulator=acc) == pytest.approx(0.3)


def test_mean_train_loss_for_epoch_averages_steps():
    history = [
        {"epoch": 1, "loss": 0.4},
        {"epoch": 1, "loss": 0.2},
        {"epoch": 2, "loss": 0.1},
    ]
    assert mean_train_loss_for_epoch(history, 1) == pytest.approx(0.3)
    assert mean_train_loss_for_epoch(history, 2) == pytest.approx(0.1)


def test_build_train_log_row_val_columns():
    row = build_train_log_row(
        1,
        0.5,
        val_geometry={"eta2_macro_balanced_perc": 10.0, "eta2_macro_balanced": 0.1},
    )
    assert row["epoch"] == 1
    assert row["val_eta2_macro_balanced_perc"] == 10.0
    assert row["val_eta2_macro_balanced"] == 0.1


def test_train_log_columns_list():
    from metrics.geometry import GEOMETRY_METRIC_KEYS

    assert "epoch" in TRAIN_LOG_COLUMNS
    assert "val_eta2_macro_balanced_perc" in TRAIN_LOG_COLUMNS
    assert "val_rankme_global" not in TRAIN_LOG_COLUMNS
