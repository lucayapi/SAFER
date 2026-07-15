"""Tests stage_bn_exports_from_bertopic_run."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_staging_path = Path(__file__).resolve().parents[1] / "bn_pipeline" / "staging_macro_transfer.py"
_spec = importlib.util.spec_from_file_location("staging_macro_transfer", _staging_path)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
stage_bn_exports_from_bertopic_run = _mod.stage_bn_exports_from_bertopic_run


def test_stage_bn_exports_from_bertopic_run(tmp_path):
    run_dir = tmp_path / "bertopic_run"
    topics = run_dir / "topics_bertopic"
    topics.mkdir(parents=True)
    transfer = run_dir / "transfer"
    transfer.mkdir(parents=True)

    n = 3
    pd.DataFrame(
        {
            "doc_idx": [0, 1, 2],
            "macro": ["A0", "A1", "B"],
            "topic_id": [0, 0, 1],
            "gamma": [0.9, 0.8, 0.7],
        }
    ).to_csv(topics / "assignments.csv", index=False)
    pd.DataFrame(
        {
            "macro": ["A0"],
            "topic_id": [0],
            "theme_label": ["thème test"],
            "top_words": ["a b"],
            "n_units": [2],
        }
    ).to_csv(topics / "themes_by_macro.csv", index=False)
    pd.DataFrame(
        {
            "accident_id": ["a1", "a1", "a2"],
            "doc_id": [0, 1, 2],
            "pred_macro": ["A0", "A1", "B"],
            "confidence": [0.8, 0.7, 0.9],
            "prob_A0": [0.7, 0.1, 0.1],
            "prob_A1": [0.1, 0.7, 0.1],
            "prob_B": [0.1, 0.1, 0.7],
            "prob_C": [0.1, 0.1, 0.1],
        }
    ).to_csv(transfer / "target_macro_predictions.csv", index=False)

    exports = stage_bn_exports_from_bertopic_run(
        run_dir,
        "batch_triplet",
        corpus_id="metallurgie",
        output_dir=tmp_path / "bn_out",
    )
    assert (exports / "metadata_with_predictions.csv").is_file()
    assert (exports / "macro_topic_assignments.csv").is_file()
    assert (exports / "themes_by_macro.csv").is_file()
    pt = np.load(exports / "pt_y_target.npy")
    assert pt.shape == (n, 4)
