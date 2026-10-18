"""Guardrails for the GOLD v3 role-transfer campaign."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import role_gold_campaign as campaign


def test_array_mapping_is_complete_and_unique():
    tasks = [campaign.task_spec(i) for i in range(54)]
    assert len(set(tasks)) == 54
    assert set(tasks) == {(corpus, seed, method)
                          for corpus in campaign.CORPORA
                          for seed in campaign.SEEDS
                          for method in campaign.METHODS}


def test_selection_uses_only_source_train_and_validation():
    frames, splits = {}, {}
    for corpus in campaign.CORPORA:
        frames[corpus] = pd.DataFrame({"corpus": [corpus] * 3,
                                        "accident_id": [f"{corpus}-{part}" for part in ("train", "val", "test")],
                                        "label_id": [0, 0, 0]})
        splits[corpus] = {part: [f"{corpus}-{part}"] for part in ("train", "val", "test")}
    train, val, targets = campaign.selection("single_source", "btp", frames, splits)
    assert set(train.corpus) == {"btp"}
    assert set(val.accident_id) == {"btp-val"}
    assert len(targets) == 5 and "btp" not in targets
    assert all(set(frame.accident_id) == {f"{target}-test"} for target, frame in targets.items())
    train, val, targets = campaign.selection("lodo", "btp", frames, splits)
    assert set(train.corpus) == set(campaign.CORPORA) - {"btp"}
    assert set(val.corpus) == set(campaign.CORPORA) - {"btp"}
    assert set(targets) == {"btp"}
    assert set(targets["btp"].accident_id) == {"btp-test"}


def test_scores_use_five_roles_even_if_one_is_never_predicted():
    y = np.array([0, 1, 2, 3, 4])
    predicted = np.array([0, 1, 2, 3, 0])
    result = campaign.scores(y, predicted)
    assert result["balanced_accuracy"] == 0.8
    assert result["recall_by_role"]["AUTRE"] == 0.0
    assert np.asarray(result["confusion_matrix"]).shape == (5, 5)


def test_logistic_selection_uses_only_supplied_validation():
    train = np.eye(5, dtype=np.float32).repeat(4, axis=0)
    labels = np.repeat(np.arange(5), 4)
    val = np.eye(5, dtype=np.float32)
    score, c, model = campaign.select_lr(train, labels, val, np.arange(5), 42)
    pred, probabilities = campaign.predict_lr(model, val)
    assert c in campaign.CS and score == 1.0
    assert pred.tolist() == list(range(5))
    assert probabilities.shape == (5, 5)


def test_completion_is_bound_to_campaign_and_prediction_hash(tmp_path: Path):
    root = tmp_path / "campaign.json"
    root.write_text("{}", encoding="utf-8")
    job = tmp_path / "job"
    job.mkdir()
    pred = job / "predictions_btp.csv"
    pred.write_text("x\n", encoding="utf-8")
    classifier = job / "classifier.joblib"
    classifier.write_bytes(b"test")
    campaign.write_json(job / "complete.json", {"campaign_sha256": campaign.digest(root),
        "classifier_sha256": campaign.digest(classifier),
        "targets": {"btp": {"predictions_sha256": campaign.digest(pred)}}})
    assert campaign.complete_and_verified(job, ["btp"], root, checkpoint=False)
    pred.write_text("y\n", encoding="utf-8")
    assert not campaign.complete_and_verified(job, ["btp"], root, checkpoint=False)


def test_pruning_requires_verified_predictions_and_allows_summary(tmp_path: Path):
    root = tmp_path / "campaign.json"
    root.write_text("{}", encoding="utf-8")
    job = tmp_path / "job"
    model_dir = job / "best_model"
    model_dir.mkdir(parents=True)
    model = model_dir / "hf_model.bin"
    model.write_bytes(b"selected weights")
    pred = job / "predictions_btp.csv"
    pred.write_text("prediction\nA0\n", encoding="utf-8")
    pred_hash = campaign.digest(pred)
    campaign.write_json(job / "metrics_btp.json", {"predictions_sha256": pred_hash})
    campaign.write_json(job / "complete.json", {
        "campaign_sha256": campaign.digest(root),
        "checkpoint_files_sha256": {"hf_model.bin": campaign.digest(model)},
        "targets": {"btp": {"predictions_sha256": pred_hash}},
    })
    campaign.prune_verified_checkpoint(job, ["btp"], root)
    assert not model_dir.exists()
    assert campaign.read_json(job / "complete.json")["checkpoint_retention"] == "predictions_only"
    assert campaign.complete_and_verified(job, ["btp"], root, checkpoint=True)
    pred.write_text("prediction\nB\n", encoding="utf-8")
    assert not campaign.complete_and_verified(job, ["btp"], root, checkpoint=True)


def test_pruning_refuses_missing_metrics(tmp_path: Path):
    root = tmp_path / "campaign.json"
    root.write_text("{}", encoding="utf-8")
    job = tmp_path / "job"
    model_dir = job / "best_model"
    model_dir.mkdir(parents=True)
    model = model_dir / "hf_model.bin"
    model.write_bytes(b"selected weights")
    pred = job / "predictions_btp.csv"
    pred.write_text("prediction\nA0\n", encoding="utf-8")
    campaign.write_json(job / "complete.json", {
        "campaign_sha256": campaign.digest(root),
        "checkpoint_files_sha256": {"hf_model.bin": campaign.digest(model)},
        "targets": {"btp": {"predictions_sha256": campaign.digest(pred)}},
    })
    assert not campaign.complete_and_verified(job, ["btp"], root, checkpoint=True)
    try:
        campaign.prune_verified_checkpoint(job, ["btp"], root)
    except ValueError:
        pass
    else:
        raise AssertionError("Pruning should reject missing metrics")
    assert model.exists()
