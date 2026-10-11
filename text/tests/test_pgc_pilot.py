"""Focused invariants for the source-only graph-consistency pilot."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from pgc_pilot.core import accident_batches, pgc_loss
from safer_core.kfold_eval import group_kfold_splits
from scripts.run_pgc_pilot import analyze, prediction_alignment_key, select_lambda


def test_grouped_batches_cover_each_unit_once_and_keep_accidents_distinct():
    groups = np.array(["long"] * 13 + ["a"] * 3 + ["b"] * 2 + ["c"] * 4)
    chosen = np.arange(len(groups))
    batches = accident_batches(chosen, groups, batch_size=8, chunk_size=3, seed=42)
    assert sorted(i for batch in batches for i in batch) == chosen.tolist()
    assert all(len(batch) <= 8 for batch in batches)
    assert all(max(Counter(groups[batch]).values()) <= 3 for batch in batches)
    assert batches == accident_batches(chosen, groups, batch_size=8, chunk_size=3, seed=42)


def test_graph_loss_aligned_and_reversed_directions():
    same = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]], requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1])
    accidents = ["a", "a", "b", "b"]
    loss, pairs = pgc_loss(same, labels, accidents)
    assert pairs["A0->A1"] == 1
    assert pairs["A0->B"] == 0
    assert loss.item() == pytest.approx(0., abs=1e-6)
    loss.backward()
    reversed_z = torch.tensor([[1., 0.], [0., 1.], [0., 1.], [1., 0.]])
    reversed_loss, _ = pgc_loss(reversed_z, labels, accidents)
    assert reversed_loss.item() == pytest.approx(2., abs=1e-6)


def test_zero_lambda_matches_cross_entropy_value_and_gradient():
    z = torch.randn(8, 4, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    accident_ids = ["a"] * 4 + ["b"] * 4
    head = torch.nn.Linear(4, 4)
    weights = torch.tensor([1., 2., 3., 4.])
    ce = F.cross_entropy(head(z), labels, weight=weights)
    ce_grad = torch.autograd.grad(ce, [z, head.weight, head.bias], retain_graph=True)
    graph, _ = pgc_loss(z, labels, accident_ids)
    pilot_grad = torch.autograd.grad(ce + 0.0 * graph, [z, head.weight, head.bias])
    assert all(torch.equal(a, b) for a, b in zip(ce_grad, pilot_grad))


def test_group_folds_never_split_an_accident():
    groups = np.repeat(np.array([f"accident_{i}" for i in range(20)]), 4)
    for train, validation in group_kfold_splits(groups, 3, 42):
        assert set(groups[train]).isdisjoint(set(groups[validation]))


def test_prediction_pairing_rejects_duplicate_and_different_units():
    frame = pd.DataFrame({"row_id": [0, 1], "accident_id": ["a", "b"], "true_id": [0, 1]})
    reference = prediction_alignment_key(frame)
    assert reference.equals(prediction_alignment_key(frame.copy()))
    changed = frame.copy()
    changed.loc[1, "true_id"] = 2
    assert not reference.equals(prediction_alignment_key(changed))
    duplicated = frame.copy()
    duplicated.loc[1, "row_id"] = 0
    with pytest.raises(ValueError, match="Duplicate row_id"):
        prediction_alignment_key(duplicated)


def test_selection_uses_only_completed_source_folds_and_tiebreaks_to_zero(tmp_path):
    cfg = {"lambda_grid": [0., .01], "n_folds": 3}
    for index, lam in enumerate(cfg["lambda_grid"]):
        for fold in range(3):
            out = tmp_path / "cv" / f"lambda_{index}_fold_{fold}"
            out.mkdir(parents=True)
            (out / "result.json").write_text(json.dumps({"lambda_index": index, "lambda": lam,
                "fold": fold, "balanced_accuracy": .8}), encoding="utf-8")
    select_lambda(cfg, tmp_path)
    selected = json.loads((tmp_path / "selection.json").read_text(encoding="utf-8"))
    assert selected["selected_lambda"] == 0.
    assert selected["no_source_signal"] is True
    improved = tmp_path / "cv/lambda_1_fold_2/result.json"
    improved.write_text(json.dumps({"lambda_index": 1, "lambda": .01,
        "fold": 2, "balanced_accuracy": .83}), encoding="utf-8")
    select_lambda(cfg, tmp_path)
    assert json.loads((tmp_path / "selection.json").read_text(encoding="utf-8"))["selected_lambda"] == .01
    improved.unlink()
    with pytest.raises(FileNotFoundError):
        select_lambda(cfg, tmp_path)


def test_analysis_uses_paired_target_units_and_reports_positive_delta(tmp_path):
    cfg = {"targets": ["metallurgie", "caou", "nicollin"], "n_seeds": 2,
           "n_bootstrap": 100, "bootstrap_seed": 2027}
    manifest = {"final_seeds": [11, 12]}
    (tmp_path / "selection.json").write_text(json.dumps({"selected_lambda": .01}), encoding="utf-8")
    for target in cfg["targets"]:
        for condition in ("ce", "ce_pgc"):
            for seed in range(2):
                out = tmp_path / "final" / condition / f"seed_{seed}" / "predictions"
                out.mkdir(parents=True, exist_ok=True)
                (out.parent / "run_manifest.json").write_text(json.dumps({
                    "condition": condition, "seed_index": seed, "training_seed": manifest["final_seeds"][seed],
                    "lambda": 0. if condition == "ce" else .01,
                    "target_sha256": {target_id: f"hash_{target_id}" for target_id in cfg["targets"]},
                }), encoding="utf-8")
                true = np.tile(np.arange(4), 8)
                predicted = true.copy()
                if condition == "ce":
                    predicted[true == 1] = 0
                frame = pd.DataFrame({"row_id": np.arange(len(true)),
                    "accident_id": np.repeat([f"a{i}" for i in range(8)], 4),
                    "true_id": true, "pred_id": predicted,
                    "true_macro": np.asarray(("A0", "A1", "B", "C"))[true],
                    "pred_macro": np.asarray(("A0", "A1", "B", "C"))[predicted]})
                for role in ("A0", "A1", "B", "C"):
                    frame[f"prob_{role}"] = .25
                frame.to_csv(out / f"predictions_{target}.csv", index=False)
    analyze(cfg, tmp_path, manifest)
    contrasts = pd.read_csv(tmp_path / "paired_differences.csv")
    assert len(contrasts) == 4
    assert (contrasts["mean_delta_ba"] > 0).all()
    assert (tmp_path / "ood_summary.csv").is_file()
    altered = tmp_path / "final/ce_pgc/seed_1/predictions/predictions_nicollin.csv"
    frame = pd.read_csv(altered)
    frame.loc[0, "row_id"] = 99
    frame.to_csv(altered, index=False)
    with pytest.raises(ValueError, match="not paired"):
        analyze(cfg, tmp_path, manifest)
