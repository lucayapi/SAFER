"""Tests CV supervised_macro_ft — class_weight (pas d'oversampling)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from scgm_text.dataset_text_raw import TextRawDataset
from supervised_macro_ft.cv import run_group_kfold_cv


@pytest.fixture
def tiny_btp_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "sentence": [f"s{i}" for i in range(12)],
            "pred_label": ["A0", "A0", "A0", "A1", "A1", "B", "B", "C"] * 1
            + ["A0", "A1", "B", "C"],
            "pred_ok": [True] * 12,
            "accident_id": list(range(12)),
        }
    )
    path = tmp_path / "btp.csv"
    df.to_csv(path, index=False)
    return path


@patch(
    "supervised_macro_ft.cv.evaluate_loader",
    return_value={"loss": 0.5, "accuracy": 1.0, "macro_f1": 1.0, "balanced_accuracy": 1.0},
)
@patch("supervised_macro_ft.cv.fit_model")
def test_cv_class_weight_keeps_train_size(mock_fit, mock_evaluate, tiny_btp_csv):
    mock_fit.return_value = (
        {},
        {
            "epoch": 1,
            "val_loss": 0.4,
            "val_accuracy": 0.9,
            "val_macro_f1": 0.9,
            "val_balanced_accuracy": 0.9,
        },
        [],
    )

    ds = TextRawDataset(str(tiny_btp_csv))
    hidden = np.random.randn(len(ds), 16).astype(np.float32)
    model_cfg = {
        "backbone_name": "__test_dummy__",
        "backbone_trainable": False,
        "cache_backbone_embeddings": True,
        "projection": "mlp_sklearn",
        "hiddim": 128,
        "n_classes": 4,
        "oversampling": False,
        "class_weight": "balanced",
        "max_seq_length": 16,
    }
    train_cfg = {"batch_size": 4, "epochs": 1, "seed": 0}

    fold_rows, _, _ = run_group_kfold_cv(
        ds,
        MagicMock(),
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        n_folds=2,
        seed=0,
        device=torch.device("cpu"),
        backbone_hidden=hidden,
        save_fold_checkpoints=False,
    )
    assert fold_rows
    assert fold_rows[0]["n_train"] == fold_rows[0]["n_train_raw"]
    assert fold_rows[0]["n_inner_train"] == fold_rows[0]["n_train_raw"]
    assert fold_rows[0]["n_inner_val"] > 0
    assert fold_rows[0]["n_outer_val"] == fold_rows[0]["n_val"]


def test_cv_rejects_oversampling(tiny_btp_csv):
    ds = TextRawDataset(str(tiny_btp_csv))
    model_cfg = {
        "backbone_name": "__test_dummy__",
        "backbone_trainable": True,
        "projection": "mlp_sklearn",
        "hiddim": 128,
        "n_classes": 4,
        "oversampling": True,
        "class_weight": "balanced",
        "max_seq_length": 16,
    }
    with pytest.raises(ValueError, match="oversampling"):
        run_group_kfold_cv(
            ds,
            MagicMock(),
            model_cfg=model_cfg,
            train_cfg={"batch_size": 4, "epochs": 1, "seed": 0},
            n_folds=2,
            seed=0,
            device=torch.device("cpu"),
            backbone_hidden=None,
            save_fold_checkpoints=False,
        )
