"""Tests log train_log.csv unifié (SupCon / Batch Triplet)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

pytest.importorskip("datasets")

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.st_common import ContrastiveEpochCallback
from contrastive_methods.training_log import (
    TRAIN_LOG_COLUMNS,
    build_train_log_row,
    mean_train_loss_for_epoch,
)


def test_mean_train_loss_for_epoch_averages_steps():
    history = [
        {"epoch": 1, "loss": 0.4},
        {"epoch": 1, "loss": 0.2},
        {"epoch": 2, "loss": 0.1},
    ]
    assert mean_train_loss_for_epoch(history, 1) == 0.3
    assert mean_train_loss_for_epoch(history, 2) == 0.1


def test_build_train_log_row_val_columns():
    row = build_train_log_row(
        1,
        0.5,
        val_geometry={"eta2_macro_balanced_perc": 10.0, "eta2_macro_balanced": 0.1, "rankme_global": 5.0},
    )
    assert row["epoch"] == 1
    assert row["val_eta2_macro_balanced_perc"] == 10.0
    assert row["val_eta2_macro_balanced"] == 0.1


@patch("contrastive_methods.st_common.evaluate_st_val_geometry")
def test_contrastive_epoch_callback_writes_standard_columns(mock_eval):
    mock_eval.return_value = {
        "eta2_macro_balanced_perc": 12.0,
        "eta2_macro_balanced": 0.12,
        "rankme_global": 8.0,
        "c1_global": 0.3,
        "c10_global": 0.5,
    }
    log_rows = []
    val_df = pd.DataFrame(
        {
            "sentence": ["a", "b"],
            "pred_label": ["A0", "A1"],
            "label_id": [0, 1],
        }
    )
    cfg = ContrastiveConfig(method_name="supcon")
    model = MagicMock()
    cb = ContrastiveEpochCallback(
        model,
        val_df,
        "sentence",
        cfg,
        Path("/tmp/best"),
        log_rows,
        use_val_geometry=True,
    )
    state = MagicMock(
        epoch=1,
        log_history=[{"epoch": 1, "loss": 0.6}, {"epoch": 1, "loss": 0.4}],
    )
    cb.on_epoch_end(None, state, None)
    assert len(log_rows) == 1
    assert log_rows[0]["train_loss"] == 0.5
    assert log_rows[0]["val_eta2_macro_balanced_perc"] == 12.0
    assert "grad_norm" not in log_rows[0]
    assert "step" not in log_rows[0]
    for col in ("epoch", "train_loss", "val_eta2_macro_balanced_perc"):
        assert col in log_rows[0]


@patch("contrastive_methods.st_common.evaluate_st_val_geometry")
def test_contrastive_epoch_callback_final_fit_train_only(mock_eval):
    log_rows = []
    cfg = ContrastiveConfig(method_name="batch_triplet", final_fit_full_data=True)
    cb = ContrastiveEpochCallback(
        MagicMock(),
        pd.DataFrame(),
        "sentence",
        cfg,
        Path("/tmp/best"),
        log_rows,
        use_val_geometry=False,
    )
    state = MagicMock(epoch=1, log_history=[{"epoch": 1, "loss": 0.25}])
    cb.on_epoch_end(None, state, None)
    mock_eval.assert_not_called()
    assert log_rows[0]["train_loss"] == 0.25
    assert log_rows[0].get("val_eta2_macro_balanced_perc") is None


def test_train_log_columns_list():
    from metrics.geometry import GEOMETRY_METRIC_KEYS

    assert "epoch" in TRAIN_LOG_COLUMNS
    assert "val_c10_global" in TRAIN_LOG_COLUMNS
    assert len([c for c in TRAIN_LOG_COLUMNS if c.startswith("val_") and c != "val_loss"]) == len(
        GEOMETRY_METRIC_KEYS
    )
