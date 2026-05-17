"""Tests K-fold train simple contrastif (mock runner)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.kfold_train import run_kfold_loop
from contrastive_methods.results import TrainingResult


def test_run_kfold_loop_mock_two_folds(tmp_path):
    cfg = ContrastiveConfig(
        method_name="batch_triplet",
        dataset_path=TEXT_ROOT / "dataset/data_btp.csv",
        output_dir=str(tmp_path / "run"),
        n_folds=2,
        seed=42,
    )

    def fake_runner(fold_cfg: ContrastiveConfig) -> TrainingResult:
        fold_id = Path(fold_cfg.output_dir).name.split("_")[-1]
        score = 10.0 + float(fold_id)
        return TrainingResult(
            embeddings_path=Path(fold_cfg.output_dir) / "emb.csv",
            output_root=Path(fold_cfg.output_dir),
            val_geometry={"delta_macro_pct": score, "eta2_macro_balanced": score / 100.0},
            best_delta_macro_pct=score,
        )

    splits = [(np.array([0, 1]), np.array([2])), (np.array([2]), np.array([0, 1]))]

    with patch("contrastive_methods.kfold_train.prepare_text_dataset") as mock_ds:
        with patch("contrastive_methods.kfold_train.get_group_kfold_splits", return_value=splits):
            mock_ds.return_value = MagicMock()
            fold_rows, agg = run_kfold_loop(cfg, fake_runner, save_tables=True)
    assert "test_delta_macro_pct" not in fold_rows[0]

    assert len(fold_rows) == 2
    assert agg["n_folds"] == 2
    assert abs(agg["mean_delta_macro_pct"] - 10.5) < 1e-6
    summary = tmp_path / "run" / "metrics" / "kfold_summary.csv"
    assert summary.is_file()


def test_methods_yaml_n_folds_is_five():
    from contrastive_methods.config import load_contrastive_config

    cfg = load_contrastive_config(
        "batch_triplet",
        TEXT_ROOT / "configs/methods/batch_triplet.yaml",
    )
    assert cfg.n_folds == 5
    assert cfg.selection_metric == "delta_macro_pct"
