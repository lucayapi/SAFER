#!/usr/bin/env python3
"""Audit and summarize completed reference-unit role-classification LODO runs.

Only completed, hashed runs are treated as observations. Empty CE directories
and methods absent from the campaign are reported as unavailable, not as zeros.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

TEXT = Path(__file__).resolve().parents[1]
ROOT = TEXT.parent
sys.path.insert(0, str(TEXT))

import role_qwen3_lodo_campaign as campaign  # noqa: E402

CORPORA = tuple(campaign.CORPORA)
ROLES = tuple(campaign.ROLES)
SEEDS = tuple(campaign.SEEDS)
OBSERVED = ("tfidf_lr", "qwen3_frozen_lr", "supcon", "softtriple")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mean_sd(values: list[float]) -> tuple[float, float | None]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None


def independent_metrics(truth: list[str], predicted: list[str]) -> tuple[float, float, list[list[int]]]:
    indices = {role: index for index, role in enumerate(ROLES)}
    matrix = [[0] * len(ROLES) for _ in ROLES]
    for expected, actual in zip(truth, predicted, strict=True):
        if expected not in indices or actual not in indices:
            raise ValueError(f"Unknown role: {expected}/{actual}")
        matrix[indices[expected]][indices[actual]] += 1
    supports = [sum(row) for row in matrix]
    if any(count == 0 for count in supports):
        raise ValueError("A test corpus lacks a reference role")
    pred_counts = [sum(row[index] for row in matrix) for index in range(len(ROLES))]
    recall = [matrix[index][index] / supports[index] for index in range(len(ROLES))]
    f1 = [2 * matrix[index][index] / (supports[index] + pred_counts[index])
          for index in range(len(ROLES))]
    return statistics.mean(recall), statistics.mean(f1), matrix


def checked_run(run: Path, target: str, method: str, seed: int | None,
                campaign_sha: str, expected_frame) -> tuple[dict, list[list[int]]]:
    marker = json.loads((run / "complete.json").read_text(encoding="utf-8"))
    if marker.get("campaign_sha256") != campaign_sha:
        raise ValueError(f"Campaign binding differs: {run}")
    expected_files = {"predictions.csv", "metrics.json", "history.json"}
    if set(marker.get("files_sha256", {})) != expected_files:
        raise ValueError(f"Missing audit files: {run}")
    for name, expected in marker["files_sha256"].items():
        if digest(run / name) != expected:
            raise ValueError(f"Changed run file: {run / name}")
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    if (metrics.get("target"), metrics.get("method"), metrics.get("seed")) != (target, method, seed):
        raise ValueError(f"Wrong run identity: {run}")
    if metrics.get("predictions_sha256") != digest(run / "predictions.csv"):
        raise ValueError(f"Prediction hash differs: {run}")

    import pandas as pd
    predictions = pd.read_csv(run / "predictions.csv", dtype={
        "accident_id": str, "doc_id": str, "fact_id": str})
    keys = ("corpus", "accident_id", "doc_id", "fact_id", "pred_label")
    if len(predictions) != len(expected_frame) or any(key not in predictions for key in keys):
        raise ValueError(f"Wrong target population: {run}")
    for key in keys:
        if predictions[key].astype(str).tolist() != expected_frame[key].astype(str).tolist():
            raise ValueError(f"Test unit order or label differs: {run}/{key}")
    if predictions[list(keys[:-1])].duplicated().any():
        raise ValueError(f"Duplicate test unit: {run}")
    probability_columns = [f"prob_{role}" for role in ROLES]
    if not all(column in predictions for column in probability_columns):
        raise ValueError(f"Missing role probabilities: {run}")
    probs = predictions[probability_columns].to_numpy(dtype=float)
    import numpy as np
    if (not np.isfinite(probs).all() or
            not np.allclose(probs.sum(axis=1), 1.0, atol=1e-3) or
            (probs < -1e-6).any() or (probs > 1 + 1e-6).any()):
        raise ValueError(f"Invalid role probabilities: {run}")
    predicted = predictions.prediction.astype(str).tolist()
    if any(ROLES[index] != actual for index, actual in zip(probs.argmax(axis=1), predicted)):
        raise ValueError(f"Prediction/probability mismatch: {run}")
    ba, macro_f1, matrix = independent_metrics(predictions.pred_label.astype(str).tolist(), predicted)
    if (metrics.get("n_units") != len(predictions) or
            metrics.get("confusion_matrix") != matrix or
            not math.isclose(metrics["balanced_accuracy"], ba, abs_tol=1e-10) or
            not math.isclose(metrics["macro_f1"], macro_f1, abs_tol=1e-10)):
        raise ValueError(f"Metrics do not reproduce predictions: {run}")
    selected = metrics["selection"]
    if (set(selected["source_ba"]) != set(CORPORA) - {target} or
            not math.isclose(selected["source_macro_ba"],
                             statistics.mean(selected["source_ba"].values()), abs_tol=1e-10)):
        raise ValueError(f"Source validation selection includes target or has wrong mean: {run}")
    if seed is not None and selected["epoch"] not in (1, 2, 3):
        raise ValueError(f"Wrong selected epoch: {run}")
    return {"target": target, "method": method, "seed": "" if seed is None else seed,
            "n_units": len(predictions), "balanced_accuracy": ba, "macro_f1": macro_f1,
            "selection_epoch": "" if seed is None else selected["epoch"]}, matrix


def summarize(rows: list[dict], matrices: dict[tuple[str, str, int | None], list[list[int]]]) -> dict:
    methods = {}
    for method in OBSERVED:
        seeds = (None,) if method.endswith("_lr") else SEEDS
        by_seed = []
        by_target = {}
        for target in CORPORA:
            group = [row for row in rows if row["target"] == target and row["method"] == method]
            if len(group) != len(seeds):
                raise ValueError(f"Incomplete method/target: {method}/{target}")
            by_target[target] = {
                "balanced_accuracy_mean": mean_sd([row["balanced_accuracy"] for row in group])[0],
                "balanced_accuracy_sd": mean_sd([row["balanced_accuracy"] for row in group])[1],
                "macro_f1_mean": mean_sd([row["macro_f1"] for row in group])[0]}
        for seed in seeds:
            group = [row for row in rows if row["method"] == method and
                     row["seed"] == ("" if seed is None else seed)]
            if len(group) != len(CORPORA):
                raise ValueError(f"Incomplete seed: {method}/{seed}")
            by_seed.append({"seed": seed,
                "macro_ba": statistics.mean(row["balanced_accuracy"] for row in group),
                "macro_f1": statistics.mean(row["macro_f1"] for row in group),
                "worst_ba": min(row["balanced_accuracy"] for row in group),
                "worst_target": min(group, key=lambda row: row["balanced_accuracy"])["target"]})
        methods[method] = {"n_seeds": len(seeds), "by_target": by_target, "by_seed": by_seed,
            "macro_ba_mean": mean_sd([item["macro_ba"] for item in by_seed])[0],
            "macro_ba_sd": mean_sd([item["macro_ba"] for item in by_seed])[1],
            "macro_f1_mean": mean_sd([item["macro_f1"] for item in by_seed])[0],
            "macro_f1_sd": mean_sd([item["macro_f1"] for item in by_seed])[1],
            "worst_ba_mean": mean_sd([item["worst_ba"] for item in by_seed])[0],
            "worst_ba_sd": mean_sd([item["worst_ba"] for item in by_seed])[1]}
    normalized = []
    for target in CORPORA:
        for seed in SEEDS:
            matrix = matrices[target, "softtriple", seed]
            normalized.append([[value / sum(row) for value in row] for row in matrix])
    confusion = [[statistics.mean(matrix[i][j] for matrix in normalized)
                  for j in range(len(ROLES))] for i in range(len(ROLES))]
    if any(not math.isclose(sum(row), 1.0, abs_tol=1e-12) for row in confusion):
        raise ValueError("Row-normalized confusion matrix does not sum to one")
    return {"methods": methods, "softtriple_confusion": {
        "roles": ROLES, "n_domain_seed_matrices": len(normalized),
        "weighting": "row normalize each target/seed confusion matrix, then equal mean over six targets and three seeds",
        "values": confusion}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path,
                        default=TEXT / "safer_role_qwen3_narrative_lodo_3epoch_v1")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/role_qwen3_narrative_lodo_3epoch_v1_verified.csv")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    binding_path = run_root / "campaign.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if (binding.get("version") != "role_qwen3_narrative_lodo_3epoch_v1" or
            binding.get("epochs") != 3 or binding.get("seeds") != list(SEEDS) or
            binding.get("methods") != list(campaign.METHODS) or
            binding.get("roles") != list(ROLES)):
        raise ValueError("Unexpected role campaign configuration")
    prepared = campaign.bundle() / "manifest.json"
    if digest(prepared) != binding["signatures"]["bundle_manifest_sha256"]:
        raise ValueError("Role campaign and SaT test bundle differ")
    frames, _, signatures = campaign.load_data(check_qwen=False)
    for key in ("bundle_manifest_sha256", "clean_manifest_sha256", "clean_csv_sha256",
                "split_sha256", "original_csv_sha256"):
        if signatures[key] != binding["signatures"][key]:
            raise ValueError(f"Role campaign source changed: {key}")
    rows = []
    matrices = {}
    campaign_sha = digest(binding_path)
    expected_markers = set()
    for target in CORPORA:
        for method in OBSERVED:
            seeds = (None,) if method.endswith("_lr") else SEEDS
            for seed in seeds:
                run = (run_root / "baselines" / target / method if seed is None else
                       run_root / "trained" / target / method / f"seed_{seed}")
                expected_markers.add((run / "complete.json").resolve())
                row, matrix = checked_run(run, target, method, seed, campaign_sha, frames[target])
                rows.append(row)
                matrices[target, method, seed] = matrix
        for seed in SEEDS:
            ce_run = run_root / "trained" / target / "ce" / f"seed_{seed}"
            if (ce_run / "complete.json").exists() or (ce_run / "predictions.csv").exists():
                raise ValueError(f"CE is no longer unavailable: {ce_run}")
    actual_markers = {path.resolve() for group in ("baselines", "trained")
                      for path in (run_root / group).rglob("complete.json")}
    if actual_markers != expected_markers:
        raise ValueError("Completed campaign runs differ from the expected 48")
    if len(rows) != 48:
        raise ValueError("Expected exactly 48 completed observed runs")
    summary = summarize(rows, matrices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".json").write_text(json.dumps({
        "campaign_sha256": campaign_sha,
        "prepared_manifest_sha256": digest(prepared),
        "output_sha256": digest(args.output),
        "n_verified_runs": len(rows),
        "unavailable": {"ce": "18 empty seed directories; no complete markers",
                        "triplet": "not a method in this campaign"},
        "summary": summary}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n_verified_runs": len(rows), "methods": {
        method: {key: value for key, value in info.items() if key in
                 ("macro_ba_mean", "macro_ba_sd", "worst_ba_mean", "worst_ba_sd", "macro_f1_mean")}
        for method, info in summary["methods"].items()}}, indent=2))


if __name__ == "__main__":
    main()
