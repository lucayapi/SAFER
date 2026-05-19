"""Tests fit final + éval test après K-fold (train simple)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.results import TrainingResult


def test_run_contrastive_final_fit_and_eval_calls_eval_once(tmp_path):
    from contrastive_methods import kfold_train as kt

    cfg = ContrastiveConfig(
        method_name="batch_triplet",
        dataset_path=TEXT_ROOT / "dataset/data_btp.csv",
        output_dir=str(tmp_path / "run"),
        n_folds=5,
    )
    fake_result = TrainingResult(
        embeddings_path=tmp_path / "emb.csv",
        output_root=tmp_path / "run",
        best_eta2_macro_balanced_perc=42.0,
    )
    ckpt_dir = tmp_path / "run" / "checkpoints" / "best_model"
    ckpt_dir.mkdir(parents=True)

    with patch.object(kt, "get_contrastive_runner") as mock_runner_fn:
        with patch.object(kt, "evaluate_btp_and_test") as mock_eval:
            mock_runner_fn.return_value = MagicMock(return_value=fake_result)
            mock_eval.return_value = {"test": tmp_path / "metrics_geometry_test.csv"}
            kt.run_contrastive_final_fit_and_eval(cfg)

    mock_eval.assert_called_once()
    call_ckpt = mock_eval.call_args[0][1]
    assert call_ckpt == ckpt_dir
