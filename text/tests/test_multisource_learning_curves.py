"""Checks for feasible equal-budget design and paired analysis invariants."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.multisource_learning_curves import analyze, feasible_conditions, quotas


def test_direct_script_entrypoint_resolves_project_modules():
    text_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(text_root / "scripts" / "multisource_learning_curves.py"), "--help"],
        cwd=text_root, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_budget_feasibility_respects_smallest_source():
    available = {"btp": 42244, "metallurgie": 22456, "caou": 6157}
    for budget in (2000, 4000, 6000):
        assert len(feasible_conditions(budget, available)) == 7
    for budget in (9000, 12000):
        conditions = feasible_conditions(budget, available)
        assert len(conditions) == 6
        assert "caou" not in conditions
        assert "btp_caou" in conditions
        assert "btp_metallurgie_caou" in conditions
        for allocation in conditions.values():
            assert sum(allocation.values()) == budget
            assert max(allocation.values()) - min(allocation.values()) <= 1
    assert quotas(2000, ("btp", "metallurgie", "caou")) == {
        "btp": 667, "metallurgie": 667, "caou": 666,
    }


def test_analysis_pairs_target_units_and_detects_missing_run(tmp_path: Path):
    roles = ("A0", "A1", "B", "C")
    reference = pd.DataFrame([
        {"accident_id": str(accident), "fact_id": str(role_idx), "true_macro": role}
        for accident in range(16) for role_idx, role in enumerate(roles)
    ]).sort_values(["accident_id", "fact_id"]).reset_index(drop=True)
    reference.to_csv(tmp_path / "target_keys.csv", index=False)
    tasks = []
    source_rows = []
    for budget in (2000, 4000):
        for draw in (0, 1):
            for condition in ("btp", "btp_metallurgie"):
                source_rows.append({"budget": budget, "condition": condition,
                                    "draw": draw, "actual_units": budget, "accidents": 100})
                for seed in (101, 202):
                    out = tmp_path / "runs" / f"n{budget}" / condition / f"draw{draw}_seed{seed}"
                    (out / "predictions").mkdir(parents=True)
                    prediction = reference.copy()
                    prediction["pred_macro"] = prediction["true_macro"]
                    # The winner switches with budget; comparisons must remain budget-specific.
                    if (budget == 2000 and condition == "btp") or (
                        budget == 4000 and condition == "btp_metallurgie"
                    ):
                        prediction.loc[prediction["true_macro"] == "A1", "pred_macro"] = "A0"
                    prediction.to_csv(out / "predictions" / "predictions_nicollin.csv", index=False)
                    (out / "target_metrics.json").write_text(json.dumps({
                        "balanced_accuracy": 0.75 if (prediction["pred_macro"] != prediction["true_macro"]).any() else 1.0
                    }), encoding="utf-8")
                    (out / "done.json").write_text("{}", encoding="utf-8")
                    tasks.append({"task_id": len(tasks), "arm": "learning_curve", "budget": budget,
                                  "condition": condition, "draw": draw, "seed": seed,
                                  "source_csv": "unused.csv", "run_dir": str(out)})
    pd.DataFrame(tasks).to_csv(tmp_path / "tasks.csv", index=False)
    pd.DataFrame(source_rows).to_csv(tmp_path / "source_samples.csv", index=False)
    analyze({"bootstrap_seed": 17, "bootstrap_resamples": 50}, tmp_path)
    contrasts = pd.read_csv(tmp_path / "analysis" / "paired_accident_bootstrap.csv")
    selected = contrasts[contrasts["comparator"] == "btp"].set_index("budget")
    assert selected.loc[2000, "delta_ba"] == pytest.approx(0.25)
    assert selected.loc[4000, "delta_ba"] == pytest.approx(-0.25)
    assert (tmp_path / "analysis" / "learning_curves.pdf").is_file()

    (Path(tasks[0]["run_dir"]) / "done.json").unlink()
    with pytest.raises(FileNotFoundError, match="Task incomplete"):
        analyze({"bootstrap_seed": 17, "bootstrap_resamples": 50}, tmp_path)
