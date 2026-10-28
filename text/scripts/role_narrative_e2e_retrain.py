#!/usr/bin/env python3
"""Source-only role training with real SaT-span inference before checkpoint pruning.

This is a NEW 54-task training campaign. It cannot recover the pruned weights
of the earlier role campaign or claim that its scores come from those weights.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import compare_narrative_only_saved_runs as spans
import role_narrative_e2e_baselines as e2e
import role_qwen3_lodo_campaign as base
import role_qwen3_lodo_missing as missing

METHODS = ("ce", "supcon", "softtriple")
VERSION = "role_narrative_e2e_reduced_v1"


def paths() -> tuple[Path, Path, Path]:
    storage = Path(os.environ.get("SCRATCH", base.ROOT))
    prepared = Path(os.environ.get("SAT_NARRATIVE_PREPARED",
        base.ROOT / "annotation/lodo_sat_transfer/narrative_only_lodo_v1"))
    sat_baselines = Path(os.environ.get("SAT_NARRATIVE_BASELINE_ROOT",
        storage / "safer_sat_lodo/runs/narrative_only_lodo_v3_baselines"))
    sat_m3 = Path(os.environ.get("SAT_NARRATIVE_RUN_ROOT",
        storage / "safer_sat_lodo/runs/narrative_only_lodo_checkpoint_v1"))
    return tuple(p.resolve() for p in (prepared, sat_baselines, sat_m3))


def manifest_path() -> Path:
    return base.root() / "e2e_retrain_manifest.json"


def expected_manifest() -> dict:
    prepared, sat_baselines, sat_m3 = paths()
    parent = base.campaign()
    extension = missing.extension_manifest()
    provenance = spans.verify_provenance(prepared, sat_baselines, sat_m3)
    if (base.read(parent)["signatures"]["bundle_manifest_sha256"] !=
            provenance["prepared_manifest_sha256"]):
        raise ValueError("Role and SaT LODO bundles differ")
    return {"version": VERSION, "script_sha256": base.digest(Path(__file__)),
            "parent_campaign_sha256": base.digest(parent),
            "extension_sha256": base.digest(extension),
            "sat_provenance": provenance, "methods": list(METHODS),
            "seeds": list(base.SEEDS), "targets": list(base.CORPORA),
            "selection": "five source validation domains, unchanged original protocol",
            "target_use": "inference and final evaluation only; no target-based selection"}


def prepare() -> None:
    storage = Path(os.environ.get("SCRATCH", base.ROOT))
    previous = (storage / "safer_role_qwen3_narrative_lodo_3epoch_v1").resolve()
    if base.root() == previous:
        raise ValueError("The end-to-end retraining root must differ from the previous role run")
    if len({task(index) for index in range(54)}) != 54:
        raise ValueError("The 54 array tasks do not map to distinct target/seed/method triples")
    base.prepare(None)
    missing.extension_manifest(create=True)
    expected = expected_manifest()
    path = manifest_path()
    if path.is_file() and base.read(path) != expected:
        raise ValueError(f"Existing retrain manifest differs: {path}")
    base.write(path, expected)
    print(json.dumps({"prepared": str(path), "training_tasks": 54,
                      "model_retraining_required": True}), flush=True)


def require_manifest() -> dict:
    path = manifest_path()
    if not path.is_file():
        raise RuntimeError("Run the new end-to-end campaign preflight first")
    expected = expected_manifest()
    if base.read(path) != expected:
        raise ValueError("Retrain provenance changed since preflight")
    return expected


def task(index: int) -> tuple[str, int, str]:
    if not 0 <= index < 54:
        raise ValueError("Expected array index 0..53")
    seed = base.SEEDS[index // 18]
    target = base.CORPORA[(index // 3) % 6]
    method = METHODS[index % 3]
    return target, seed, method


def result_dir(target: str, seed: int, method: str, smoke: bool) -> Path:
    if method == "ce":
        return missing.result_dir(target, method, seed, smoke=smoke)
    return (base.root() / "smoke" / method if smoke else
            base.root() / "trained" / target / method / f"seed_{seed}")


def captured(dest: Path) -> bool:
    marker = dest / "e2e" / "complete.json"
    if not marker.is_file():
        return False
    try:
        record = base.read(marker)
        hashes = record.get("files_sha256")
        return (record.get("retrain_manifest_sha256") == base.digest(manifest_path()) and
                record.get("checkpoint_retention_policy") == "prune_after_verified_span_inference" and
                isinstance(hashes, dict) and
                set(hashes) == {"scores.csv", "predictions.csv"} and
                all(isinstance(sha, str) and base.digest(dest / "e2e" / name) == sha
                    for name, sha in hashes.items()))
    except (OSError, ValueError, TypeError):
        return False


@lru_cache(maxsize=1)
def reference_bindings() -> tuple[Path, Path]:
    parent = base.campaign()
    return parent, missing.extension_manifest()


def reference_complete(dest: Path, method: str) -> bool:
    parent, extension = reference_bindings()
    if method == "ce":
        return missing.verified_result(dest, parent, extension)
    return base.verified_done(dest, parent)


def infer_labels(model, classifier, tokenizer, texts: list[str], method: str) -> list[str]:
    import torch

    frame = pd.DataFrame({"sentence": texts, "label_id": np.zeros(len(texts), dtype=int)})
    if method == "ce":
        predicted, _ = base.old_role.ce_predictions(
            model, base.text_loader(frame, tokenizer, batch_size=16), torch.device("cuda"))
    else:
        features = base.metric_embeddings(model, frame, tokenizer, torch.device("cuda"), 32)
        predicted, _ = base.old_role.predict_lr(classifier, features)
    return [base.ROLES[int(i)] for i in predicted]


def capture(target: str, seed: int, method: str, dest: Path,
            test: pd.DataFrame, *, smoke: bool) -> None:
    import torch
    from joblib import load
    from transformers import AutoTokenizer
    from contrastive_methods.config import ContrastiveConfig
    from contrastive_methods.hf_training_common import load_contrastive_checkpoint
    from supervised_macro_ft.checkpoint_io import load_checkpoint

    checkpoint = dest / "best_model"
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Selected model missing before span inference: {checkpoint}")
    prepared, sat_baselines, sat_m3 = paths()
    reference = list(spans.load_jsonl(prepared / "folds" / f"target_{target}" / "test.jsonl"))
    keep = set(test.accident_id.astype(str)) if smoke else None
    if keep is not None:
        reference = [row for row in reference if str(row["document_id"]) in keep]
    tokenizer = AutoTokenizer.from_pretrained(base.old_qwen.backbone_path(),
                                                trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    classifier = None
    if method == "ce":
        model = load_checkpoint(checkpoint, device="cuda")
    else:
        config = ContrastiveConfig(method_name=method, dataset_path=Path("unused"),
            backbone_name=base.old_qwen.backbone_path(), input_prefix="",
            backbone_trainable=True, train_last_n_layers=None,
            cache_backbone_embeddings=False, max_seq_length=base.MAX_LENGTH,
            use_projector=True, projection="linear", hiddim=128, use_amp=True,
            encode_batch_size=32, seed=seed)
        model = load_contrastive_checkpoint(config, checkpoint, "cuda")
        classifier = load(checkpoint / "classifier.joblib")
    predictions, scores = [], []
    for segmenter in (("M2", "M3") if method == "softtriple" else ("M3",)):
        file = (sat_m3 / f"target_{target}" / "predictions.jsonl" if segmenter == "M3" else
                sat_baselines / "predictions" / target / f"{segmenter}.jsonl")
        if smoke:
            full_reference = list(spans.load_jsonl(
                prepared / "folds" / f"target_{target}" / "test.jsonl"))
            full_predictions = e2e.target_predictions(full_reference, file, segmenter)
            selected = [(row, pred) for row, pred in
                        zip(full_reference, full_predictions, strict=True)
                        if str(row["document_id"]) in keep]
            rows = [row for row, _ in selected]
            intervals = [pred for _, pred in selected]
        else:
            rows = reference
            intervals = e2e.target_predictions(rows, file, segmenter)
        texts, _, _ = e2e.prepare_segments(rows, intervals)
        if not texts:
            raise ValueError(f"No predicted spans: {target}/{segmenter}")
        labels = infer_labels(model, classifier, tokenizer, texts, method)
        score, records = e2e.score_target(target, segmenter, method, rows, intervals, labels)
        score["seed"] = seed
        scores.append(score)
        predictions.extend({"seed": seed, **record} for record in records)
        print(f"[e2e] {target}/{method}/{seed}/{segmenter}: "
              f"role F1={score['role_aware_f1']:.4f}", flush=True)
        del texts, labels
        gc.collect()
    out = dest / "e2e"
    fields = ("seed", *e2e.FIELDS)
    e2e.write_csv(out / "scores.csv", scores, fields)
    e2e.write_csv(out / "predictions.csv", predictions,
                  ("seed", *e2e.PREDICTION_FIELDS))
    base.write(out / "complete.json", {
        "retrain_manifest_sha256": base.digest(manifest_path()),
        "checkpoint_retention_policy": "prune_after_verified_span_inference",
        "files_sha256": {name: base.digest(out / name)
                         for name in ("scores.csv", "predictions.csv")}})
    del model, classifier
    gc.collect()
    torch.cuda.empty_cache()


def run_training(target: str, seed: int, method: str, *, smoke: bool = False) -> None:
    require_manifest()
    dest = result_dir(target, seed, method, smoke)
    # CE's extension marker is written by the original trainer just after our
    # capture callback. Recover that narrow interruption without refitting.
    if method == "ce" and (dest / "complete.json").is_file():
        marker = base.read(dest / "complete.json")
        if ("extension_sha256" not in marker and
                base.verified_done(dest, base.campaign()) and
                ((dest / "best_model").is_dir() or captured(dest))):
            marker["extension_sha256"] = base.digest(missing.extension_manifest())
            base.write(dest / "complete.json", marker)
    if captured(dest):
        if not reference_complete(dest, method):
            raise ValueError(f"Span result exists but reference-unit audit is incomplete: {dest}")
        missing.safe_remove_checkpoint(dest)
        print(f"[skip] verified {dest / 'e2e'}", flush=True)
        return
    # If reference-unit scoring finished but the process stopped before
    # capture, use the still-present selected checkpoint without retraining.
    if (dest / "complete.json").is_file():
        if (dest / "best_model").is_dir():
            frames, splits, _ = base.load_data(check_qwen=False)
            _, _, test = base.selection(target, frames, splits)
            if smoke:
                test = test.head(8).copy()
            capture(target, seed, method, dest, test, smoke=smoke)
            if not captured(dest) or not reference_complete(dest, method):
                raise RuntimeError("Recovered span inference did not verify")
            missing.safe_remove_checkpoint(dest)
            return
        raise RuntimeError(f"Reference run exists but selected weights are gone: {dest}")

    original = base.save_result

    def save_and_capture(folder, test, pred, probs, result_target,
                         result_method, result_seed, chosen, history):
        original(folder, test, pred, probs, result_target,
                 result_method, result_seed, chosen, history)
        capture(target, seed, method, folder, test, smoke=smoke)
        if not captured(folder):
            raise RuntimeError(f"Span inference did not verify: {folder}")

    base.save_result = save_and_capture
    try:
        if method == "ce":
            missing.train_one(target, seed, method, smoke=smoke)
        else:
            base.train_one(target, seed, method, smoke=smoke)
    finally:
        base.save_result = original
    if not captured(dest) or (dest / "best_model").exists():
        raise RuntimeError(f"End-to-end output incomplete or checkpoint not pruned: {dest}")


def smoke() -> None:
    for method in METHODS:
        run_training(base.CORPORA[0], base.SEEDS[0], method, smoke=True)
    base.write(base.root() / "e2e_smoke_complete.json", {
        "retrain_manifest_sha256": base.digest(manifest_path()),
        "methods": list(METHODS)})


def train(index: int) -> None:
    marker = base.read(base.root() / "e2e_smoke_complete.json")
    if (marker.get("retrain_manifest_sha256") != base.digest(manifest_path()) or
            marker.get("methods") != list(METHODS)):
        raise RuntimeError("Run the end-to-end retrain smoke first")
    run_training(*task(index))


def table4_rows(table: pd.DataFrame) -> list[dict]:
    panels = (
        ("Fixed classifier: SoftTriple + LR", (
            ("M2", "softtriple", "SaT, source-calibrated"),
            ("M3", "softtriple", "SaT, LoRA-adapted"))),
        ("Fixed segmenter: LoRA-adapted SaT", (
            ("M3", "ce", "Cross-entropy"),
            ("M3", "supcon", "SupCon + LR"),
            ("M3", "softtriple", "SoftTriple + LR"))),
    )
    output = []
    for panel, configurations in panels:
        for segmenter, classifier, label in configurations:
            found = table[table.segmenter.eq(segmenter) & table.classifier.eq(classifier)]
            if len(found) != 1:
                raise ValueError(f"Missing or duplicate Table 4 result: {segmenter}/{classifier}")
            row = found.iloc[0]
            output.append({"panel": panel, "configuration": label,
                           "segmenter": segmenter, "classifier": classifier,
                           "role_aware_f1": row.role_f1_mean,
                           "seed_sd": "" if pd.isna(row.role_f1_sd) else row.role_f1_sd,
                           "n_seeds": int(row.n_seeds)})
    if len(output) != 5:
        raise ValueError("Table 4 must have five displayed lines")
    return output


def summary() -> None:
    require_manifest()
    rows = []
    for target in base.CORPORA:
        for seed in base.SEEDS:
            for method in METHODS:
                dest = result_dir(target, seed, method, False)
                if not captured(dest) or not reference_complete(dest, method):
                    raise ValueError(f"Missing or changed role-aware output: {dest}")
                task_rows = pd.read_csv(dest / "e2e/scores.csv").to_dict("records")
                segments = {"M2", "M3"} if method == "softtriple" else {"M3"}
                if (len(task_rows) != len(segments) or
                        {row["segmenter"] for row in task_rows} != segments or
                        any(row["target"] != target or row["classifier"] != method or
                            int(row["seed"]) != seed or row["n_documents"] <= 0
                            for row in task_rows)):
                    raise ValueError(f"Unexpected target, seed or segmenter scores: {dest}")
                rows.extend(task_rows)
    table = pd.DataFrame(rows)
    expected = 6 * 3 * 4  # SoftTriple x M2/M3, CE and SupCon x M3
    if len(table) != expected:
        raise ValueError(f"Expected {expected} adapted target/seed/segmenter scores")
    adapted = table.groupby(["classifier", "segmenter", "seed"], as_index=False).agg(
        macro_role_f1=("role_aware_f1", "mean"), n_targets=("target", "nunique"))
    if not adapted.n_targets.eq(6).all():
        raise ValueError("Missing adapted target for a seed")
    overview = adapted.groupby(["classifier", "segmenter"], as_index=False).agg(
        role_f1_mean=("macro_role_f1", "mean"),
        role_f1_sd=("macro_role_f1", "std"), n_seeds=("seed", "nunique"))
    table_out = overview
    expected_pairs = {("softtriple", "M2"), ("softtriple", "M3"),
                      ("ce", "M3"), ("supcon", "M3")}
    if (len(table_out) != 4 or
            set(zip(table_out.classifier, table_out.segmenter)) != expected_pairs or
            not table_out.n_seeds.eq(3).all()):
        raise ValueError("Expected four configurations with three seeds each")
    output = base.root() / "e2e_results"
    e2e.write_csv(output / "adapted_by_target_seed.csv", rows,
                  ("seed", *e2e.FIELDS))
    e2e.write_csv(output / "macro_by_seed.csv", adapted.to_dict("records"),
                  tuple(adapted.columns))
    reference_rows = []
    for target in base.CORPORA:
        for seed in base.SEEDS:
            for method in METHODS:
                metrics = base.read(result_dir(target, seed, method, False) / "metrics.json")
                reference_rows.append({"target": target, "seed": seed, "method": method,
                                       "balanced_accuracy": metrics["balanced_accuracy"],
                                       "macro_f1": metrics["macro_f1"]})
    e2e.write_csv(output / "reference_unit_metrics.csv", reference_rows,
                  ("target", "seed", "method", "balanced_accuracy", "macro_f1"))
    e2e.write_csv(output / "table4_real_values.csv", table_out.to_dict("records"),
                  tuple(table_out.columns))
    display = table4_rows(table_out)
    e2e.write_csv(output / "table4_display.csv", display, tuple(display[0]))
    base.write(output / "summary.json", {
        "version": VERSION, "retrain_manifest_sha256": base.digest(manifest_path()),
        "reference_unit_metrics_sha256": base.digest(output / "reference_unit_metrics.csv"),
        "macro_by_seed_sha256": base.digest(output / "macro_by_seed.csv"),
        "table_sha256": base.digest(output / "table4_real_values.csv"),
        "display_sha256": base.digest(output / "table4_display.csv"),
        "note": "SoftTriple/M3 appears in both panels; all four configurations come from new retraining"})
    print(table_out.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "smoke", "train", "summary"))
    parser.add_argument("--task-index", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "smoke":
        smoke()
    elif args.command == "train":
        if args.task_index is None:
            parser.error("train requires --task-index 0..53")
        train(args.task_index)
    else:
        summary()


if __name__ == "__main__":
    main()
