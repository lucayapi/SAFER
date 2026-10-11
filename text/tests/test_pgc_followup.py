"""Source-selection and paired-analysis checks for the PGC follow-up."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from contrastive_methods.losses.supcon_hobbit import SupConEmbeddingLoss
from pgc_pilot.core import pgc_loss
from scripts.run_pgc_followup import analyze_family, summarize_loss_diagnostics, supcon_select


def test_zero_lambda_matches_supcon_value_and_gradient():
    z = torch.randn(8, 5, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    loss = SupConEmbeddingLoss()(z, labels)
    baseline_grad = torch.autograd.grad(loss, z, retain_graph=True)[0]
    graph, _ = pgc_loss(z, labels, ["a"] * 4 + ["b"] * 4)
    trial_grad = torch.autograd.grad(loss + 0.0 * graph, z)[0]
    assert torch.equal(baseline_grad, trial_grad)


def test_supcon_selection_source_only_and_rejects_missing_fold(tmp_path):
    cfg = {"lambda_grid": [0.0, 0.01], "n_folds": 3}
    for i, lam in enumerate(cfg["lambda_grid"]):
        for fold in range(3):
            out = tmp_path / "supcon_cv" / f"lambda_{i}_fold_{fold}"
            out.mkdir(parents=True)
            (out / "result.json").write_text(json.dumps({"lambda_index": i, "lambda": lam,
                "fold": fold, "balanced_accuracy": 0.8}), encoding="utf-8")
            pd.DataFrame([{"epoch": 1, "lambda": lam, "pgc": .5, "weighted_pgc": .5 * lam,
                "weighted_loss_ratio": .1, "weighted_z_grad_ratio": .2, "active_edges": 2}]).to_csv(
                    out / "batch_diagnostics.csv", index=False)
    supcon_select(cfg, tmp_path)
    assert json.loads((tmp_path / "supcon_selection.json").read_text())["selected_lambda"] == 0.0
    assert (tmp_path / "supcon_cv_loss_diagnostics.csv").is_file()
    (tmp_path / "supcon_cv/lambda_1_fold_2/result.json").unlink()
    with pytest.raises(FileNotFoundError):
        supcon_select(cfg, tmp_path)


def test_selected_zero_reuses_identical_supcon_baseline_predictions(tmp_path):
    cfg = {"targets": ["company"], "lambda_grid": [0., .01], "n_seeds": 2,
           "n_bootstrap": 20, "bootstrap_seed": 9}
    manifest = {"final_seeds": [11, 12]}
    (tmp_path / "supcon_selection.json").write_text(json.dumps({"selected_lambda": 0.0}))
    for seed in range(2):
        out = tmp_path / "supcon_final" / "condition_0" / f"seed_{seed}"
        (out / "predictions").mkdir(parents=True)
        (out / "done.json").write_text(json.dumps({"seed_index": seed, "lambda_index": 0,
            "training_seed": manifest["final_seeds"][seed], "lambda": 0.0,
            "target_sha256": {"company": "unchanged"}}))
        truth = np.tile(np.arange(4), 16)
        frame = pd.DataFrame({"row_id": np.arange(len(truth)),
            "accident_id": np.repeat([f"acc_{i}" for i in range(16)], 4),
            "true_id": truth, "pred_id": truth, "true_macro": np.array(["A0", "A1", "B", "C"])[truth],
            "pred_macro": np.array(["A0", "A1", "B", "C"])[truth]})
        for role in ["A0", "A1", "B", "C"]:
            frame[f"prob_{role}"] = (frame.pred_macro == role).astype(float)
        frame.to_csv(out / "predictions" / "predictions_company.csv", index=False)
        pd.DataFrame([{"epoch": 1, "lambda": 0.0, "pgc": .5, "weighted_pgc": 0.,
            "weighted_loss_ratio": 0., "weighted_z_grad_ratio": 0., "active_edges": 2}]).to_csv(
                out / "batch_diagnostics.csv", index=False)
    analyze_family(cfg, tmp_path, manifest, "supcon_final")
    contrasts = pd.read_csv(tmp_path / "supcon_final_paired_differences.csv")
    assert (contrasts.mean_delta_ba == 0).all()
    assert (contrasts.ci_low == 0).all()
    assert (contrasts.ci_high == 0).all()


def test_loss_diagnostics_reject_nonfinite_values(tmp_path):
    path = tmp_path / "run" / "batch_diagnostics.csv"
    path.parent.mkdir()
    pd.DataFrame([{"epoch": 1, "lambda": .1, "pgc": np.inf,
        "weighted_pgc": .1, "weighted_loss_ratio": .1,
        "weighted_z_grad_ratio": .1, "active_edges": 1}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Non-finite"):
        summarize_loss_diagnostics(tmp_path, tmp_path / "summary.csv")
