"""Scientific invariants for source-mixture sampling and target alignment."""

from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.multisource_dg import _aligned_predictions, _key_table, _metrics, _sample_nested


def test_direct_script_entrypoint_resolves_project_modules():
    text_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(text_root / "scripts" / "multisource_dg.py"), "--help"],
        cwd=text_root, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_whole_accident_nested_sampling():
    rows = []
    roles = ["A0", "A1", "B", "C"]
    for accident in range(20):
        for fact in range(2):
            rows.append({
                "accident_id": f"btp:{accident}", "fact_id": fact,
                "pred_label": roles[(accident + fact) % 4],
            })
    frame = pd.DataFrame(rows)
    samples = _sample_nested(frame, (8, 12, 20), np.full(4, 0.25), seed=7, candidates=16)
    groups = []
    for budget in (8, 12, 20):
        sample = samples[budget]
        assert len(sample) == budget
        assert sample.groupby("accident_id").size().eq(2).all()
        groups.append(set(sample.accident_id))
    assert groups[0] < groups[1] < groups[2]


def test_duplicate_target_keys_rejected():
    frame = pd.DataFrame({"accident_id": ["a", "a"], "fact_id": [1, 1],
                          "pred_label": ["A0", "A1"]})
    with pytest.raises(ValueError, match="Duplicate target keys"):
        _key_table(frame)


def test_target_alignment_and_role_metrics(tmp_path: Path):
    reference = pd.DataFrame({
        "accident_id": ["a", "a", "b", "b"],
        "fact_id": ["1", "2", "1", "2"],
        "true_macro": ["A0", "A1", "B", "C"],
    })
    prediction = reference.copy()
    prediction["pred_macro"] = ["A0", "B", "B", "C"]
    path = tmp_path / "predictions.csv"
    prediction.iloc[::-1].to_csv(path, index=False)
    aligned = _aligned_predictions(path, reference)
    assert aligned["pred_macro"].tolist() == ["A0", "B", "B", "C"]
    metrics = _metrics(aligned)
    assert metrics["balanced_accuracy"] == 0.75
    assert metrics["a1_recall"] == 0.0
    prediction.loc[0, "true_macro"] = "C"
    prediction.to_csv(path, index=False)
    with pytest.raises(ValueError, match="keys/labels"):
        _aligned_predictions(path, reference)
