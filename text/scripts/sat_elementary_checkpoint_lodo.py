#!/usr/bin/env python3
"""Source-selected SaT-LoRA checkpoints for elementary-unit segmentation.

This rerun uses the immutable facts-plus-exclusions training bundle from
sat_elementary_lodo.py. It follows the historical M3 checkpoint procedure:
export each 1,000-step LoRA state, select checkpoint and threshold jointly on
the five source validation domains, and evaluate the held-out target once.
The output root must be new; the final-only elementary run is never overwritten.
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
from contextlib import closing
from pathlib import Path

import torch

import rq1_final_lodo as historical
import sat_elementary_lodo as elementary
import sat_lodo_checkpoint_select as checkpoint
import sat_lodo_gold_v3 as base

VERSION = "sat_elementary_checkpoint_lodo_v2"
METHOD = "SaT+LoRA elementary units; source-selected checkpoint and threshold"
CURVE_FIELDS = ("candidate", "step", "epoch", "threshold",
                "macro_elementary_exact_f1", "macro_factual_exact_recall",
                "macro_excluded_exact_recall", "selected")
METRICS = ("elementary_exact_f1", "factual_exact_recall", "excluded_exact_recall")


def signature(bundle: Path, prepared: Path) -> dict:
    return {"version": VERSION,
            "base_bundle_manifest_sha256": base.digest(bundle / "manifest.json"),
            "elementary_manifest_sha256": base.digest(prepared / "manifest.json"),
            "elementary_data_code_sha256": elementary.code_digest(),
            "historical_checkpoint_code_sha256": base.digest(Path(historical.__file__)),
            "checkpoint_export_code_sha256": base.digest(Path(checkpoint.__file__)),
            "script_sha256": base.digest(Path(__file__)),
            "trainer_template_sha256": base.digest(base.legacy.TEMPLATE)}


def checked_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, dict]:
    bundle, prepared, root = (args.bundle.resolve(), args.prepared.resolve(),
                              args.run_root.resolve())
    elementary.check_prepared(bundle, prepared)
    base.check_bundle(bundle)
    return bundle, prepared, root, signature(bundle, prepared)


def smoke(args: argparse.Namespace) -> None:
    bundle, prepared, root, expected = checked_inputs(args)
    marker = root / "checkpoint_smoke_ok.json"
    if marker.is_file():
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ValueError("Checkpoint smoke belongs to another experiment")
        historical.adapter_hashes(base.legacy.adapter_folder(root / "smoke" / "trainer"))
        print(f"Checkpoint smoke already passed: {marker}", flush=True)
        return

    # The elementary smoke checks the dataset, pinned SaT and a loadable LoRA.
    elementary.smoke(args)
    repo = args.wtpsplit_repo.resolve()
    base.legacy.git_revision(repo)
    base.legacy.gpu_runtime(repo)
    trainer = root / "smoke" / "trainer"
    final = base.legacy.adapter_folder(trainer)
    first = trainer / "checkpoint-1"
    exported = root / "smoke" / "checkpoint_adapter"
    checkpoint.export_checkpoint_adapter(first, exported, final)
    historical.adapter_hashes(exported)

    # Verify that an interrupted optimizer run can be resumed safely.
    state = trainer / "trainer_state.json"
    if int(json.loads(state.read_text(encoding="utf-8"))["global_step"]) < 2:
        config = json.loads((root / "smoke" / "config.json").read_text(encoding="utf-8"))
        config.update({"max_steps": 2, "save_total_limit": None,
                       "resume_from_checkpoint": str(first)})
        checkpoint.run_upstream(repo, config, root / "smoke" / "resume_config.json")
    if int(json.loads(state.read_text(encoding="utf-8"))["global_step"]) != 2:
        raise AssertionError("Optimizer resume did not reach step 2")
    second = root / "smoke" / "resumed_checkpoint_adapter"
    checkpoint.export_checkpoint_adapter(trainer / "checkpoint-2", second, final)
    saved = torch.load(final / "pytorch_adapter.bin", map_location="cpu", weights_only=True)
    resumed = torch.load(second / "pytorch_adapter.bin", map_location="cpu", weights_only=True)
    if set(saved) != set(resumed) or any(not torch.equal(saved[k], resumed[k]) for k in saved):
        raise AssertionError("Resumed checkpoint and final adapter differ")
    checkpoint.write_json_atomic(marker, expected)
    print(f"Checkpoint export and resume passed: {marker}", flush=True)


def validation_curve(rows: list[dict], db, candidate: dict) -> list[dict]:
    """Score one frozen candidate on sources only, using elementary gold."""
    _, curve = elementary.select_threshold(rows, db)
    result = []
    for row in curve:
        result.append({"candidate": candidate["candidate"], "step": candidate["step"],
                       "epoch": candidate["epoch"], "threshold": row["threshold"],
                       "macro_elementary_exact_f1": row["macro_elementary_exact_f1"],
                       "macro_factual_exact_recall": row["macro_factual_exact_recall"],
                       "macro_excluded_exact_recall": row["macro_excluded_exact_recall"],
                       "selected": False})
    return result


def best_candidate(curve: list[dict]) -> dict:
    """The old M3 tie-break, with elementary F1 as the current objective."""
    if not curve:
        raise ValueError("No checkpoint/threshold candidates")
    return min(curve, key=lambda row: (-row["macro_elementary_exact_f1"],
                                       abs(row["threshold"] - base.DEFAULT),
                                       row["step"], row["threshold"]))


def bound_cache(path: Path, expected: dict) -> None:
    marker = path.with_suffix(".binding.json")
    if marker.is_file():
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Probability cache uses different weights: {path}")
    elif path.exists():
        raise ValueError(f"Unbound probability cache: {path}")
    else:
        checkpoint.write_json_atomic(marker, expected)


def infer_to_cache(adapter: Path, rows: list[dict], path: Path, expected: dict,
                   model_dir: Path, tokenizer_dir: Path) -> None:
    bound_cache(path, expected)
    with closing(base._probability_connection(path, create=True)) as db:
        keys = {(row["corpus"], row["document_id"]) for row in rows}
        existing = set(db.execute("SELECT corpus,document_id FROM probabilities"))
        if not existing.issubset(keys):
            raise ValueError(f"Unexpected narratives in {path}")
        if len(existing) < len(rows):
            model = base.load_pinned_sat(model_dir, tokenizer_dir, adapter)
            base.infer_adapted(model, db, rows)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if set(db.execute("SELECT corpus,document_id FROM probabilities")) != keys:
            raise ValueError(f"Incomplete probability cache: {path}")


def evaluate_target(bundle: Path, target: str, run: Path, selection: dict,
                    model_dir: Path, tokenizer_dir: Path, expected: dict) -> None:
    rows = base.read_jsonl(bundle / "folds" / f"target_{target}" / "test.jsonl")
    selected_adapter = run / "selected_adapter"
    cache = run / "test_probabilities.sqlite3"
    infer_to_cache(selected_adapter, rows, cache,
                   {"experiment": expected, "weights": selection["weights"],
                    "candidate": selection["candidate"], "partition": "target_test"},
                   model_dir, tokenizer_dir)
    model = base.load_pinned_sat(model_dir, tokenizer_dir, selected_adapter)
    with closing(base._probability_connection(cache)) as db:
        base.verify_native(model, db, rows, selection["threshold"])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    total = collections.Counter()
    output = run / "predictions.jsonl"
    temporary = output.with_name(output.name + ".tmp")
    with closing(base._probability_connection(cache)) as db, temporary.open(
            "w", encoding="utf-8") as stream:
        for row in rows:
            predicted = base.legacy.spans_from_probabilities(
                row["raw_text"], base.get_probs(db, row), selection["threshold"])
            total.update(elementary.score(row, predicted))
            stream.write(json.dumps({"corpus": target, "document_id": row["document_id"],
                                     "threshold": selection["threshold"],
                                     "segments": [{"start": a, "end": b} for a, b in predicted]},
                                    ensure_ascii=False) + "\n")
    temporary.replace(output)
    metrics = {"version": VERSION, "method": METHOD, "target": target,
               "n_test": len(rows), "seed": base.SEED,
               "candidate": selection["candidate"], "step": selection["step"],
               "threshold": selection["threshold"],
               "selection_sha256": base.digest(run / "selection.json"),
               "predictions_sha256": base.digest(output),
               "counts": dict(total), "metrics": elementary.rates(total),
               "completed_at_utc": base.stamp()}
    checkpoint.write_json_atomic(run / "metrics.json", metrics)
    print(json.dumps({"target": target, "candidate": selection["candidate"],
                      "threshold": selection["threshold"], **metrics["metrics"]}), flush=True)


def cleanup_full_checkpoints(run: Path, selection: dict) -> None:
    """Only remove full optimizer states after all results and weights verify."""
    metrics_path = run / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (metrics["version"] != VERSION or
            metrics["target"] != run.name.removeprefix("target_") or
            metrics["selection_sha256"] != base.digest(run / "selection.json") or
            metrics["predictions_sha256"] != base.digest(run / "predictions.jsonl") or
            historical.adapter_hashes(run / "selected_adapter") != selection["weights"]):
        raise ValueError("Cannot clean checkpoints before outputs are verified")
    trainer = run / "trainer"
    released = 0
    for folder in trainer.glob("checkpoint-*"):
        if folder.is_dir() and folder.name.removeprefix("checkpoint-").isdigit():
            released += historical.remove_checkpoint(trainer, folder)
    checkpoint.write_json_atomic(run / "storage_cleanup.json",
                                 {"released_bytes": released,
                                  "full_checkpoints_remaining": 0,
                                  "selected_weights": selection["weights"]})


def fit(args: argparse.Namespace) -> None:
    bundle, prepared, root, expected = checked_inputs(args)
    marker = root / "checkpoint_smoke_ok.json"
    if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != expected:
        raise RuntimeError("Run the checkpoint GPU smoke first")
    if not 0 <= args.task_index < len(base.CORPORA):
        raise ValueError("task-index must be 0..5")
    target = base.CORPORA[args.task_index]
    repo = args.wtpsplit_repo.resolve()
    base.legacy.git_revision(repo)
    base.legacy.gpu_runtime(repo)
    model_dir, tokenizer_dir = base.pinned_backbone(bundle)
    run = root / f"target_{target}"
    elementary.bind_run(run, expected)
    metrics_path = run / "metrics.json"
    if metrics_path.is_file():
        selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
        cleanup_full_checkpoints(run, selection)
        print(f"Already completed and verified: {target}", flush=True)
        return

    trainer = run / "trainer"
    config = base.trainer_config(bundle, target, trainer, model_dir, tokenizer_dir)
    config.update({"text_path": str(prepared / "folds" / f"target_{target}" / "training.pth"),
                   "save_strategy": "steps", "save_steps": historical.SAVE_STEPS,
                   "save_total_limit": None, "load_best_model_at_end": False})
    final = base.legacy.adapter_folder(trainer)
    if not checkpoint.complete_adapter(final):
        latest = checkpoint.latest_resumable_checkpoint(trainer)
        if latest:
            config["resume_from_checkpoint"] = str(latest)
            print(f"Resuming {target} from {latest}", flush=True)
        historical.run_trainer_with_harvest(repo, config, run / "config.json", run,
                                            base.legacy.adapter_folder(root / "smoke" / "trainer"))
    historical.harvest_checkpoints(run, trainer,
                                   base.legacy.adapter_folder(root / "smoke" / "trainer"))
    candidates = historical.candidates_from_exports(run, trainer)
    base.legacy.export_loss_history(trainer / "trainer_state.json", run / "loss_history.csv")
    checkpoint.write_json_atomic(run / "training_done.json",
                                 {"experiment": expected, "target": target,
                                  "seed": base.SEED, "checkpoint_steps":
                                  [item["step"] for item in candidates],
                                  "completed_at_utc": base.stamp()})

    source_rows = base.read_jsonl(bundle / "folds" / f"target_{target}" / "val.jsonl")
    curves = []
    adapters = {}
    for candidate in candidates:
        name, adapter = candidate["candidate"], candidate["adapter"]
        weights = historical.adapter_hashes(adapter)
        cache = run / "validation_cache" / f"{name}.sqlite3"
        infer_to_cache(adapter, source_rows, cache,
                       {"experiment": expected, "weights": weights,
                        "candidate": name, "partition": "source_validation"},
                       model_dir, tokenizer_dir)
        with closing(base._probability_connection(cache)) as db:
            curve = validation_curve(source_rows, db, candidate)
        curves.extend(curve)
        adapters[name] = adapter
        base.write_csv(run / f"source_validation_{name}.csv", curve, CURVE_FIELDS)
    chosen = best_candidate(curves)
    chosen["selected"] = True
    base.write_csv(run / "checkpoint_validation.csv", curves, CURVE_FIELDS)
    selection = {"experiment": expected, "target": target,
                 "criterion": "macro exact elementary F1 over five source domains",
                 "candidate": chosen["candidate"], "step": chosen["step"],
                 "epoch": chosen["epoch"], "threshold": chosen["threshold"],
                 "source_macro_elementary_exact_f1": chosen["macro_elementary_exact_f1"],
                 "weights": historical.adapter_hashes(adapters[chosen["candidate"]]),
                 "curve_sha256": base.digest(run / "checkpoint_validation.csv")}
    selection_path = run / "selection.json"
    if selection_path.is_file():
        if json.loads(selection_path.read_text(encoding="utf-8")) != selection:
            raise ValueError("Source selection changed on resume")
    else:
        checkpoint.write_json_atomic(selection_path, selection)
    historical.persist_selected_adapter(adapters[chosen["candidate"]],
                                        run / "selected_adapter", selection["weights"])
    evaluate_target(bundle, target, run, selection, model_dir, tokenizer_dir, expected)
    cleanup_full_checkpoints(run, selection)


def summarize(args: argparse.Namespace) -> None:
    bundle, prepared, root, expected = checked_inputs(args)
    manifest = base.check_bundle(bundle)
    rows = []
    for target in base.CORPORA:
        run = root / f"target_{target}"
        if json.loads((run / "binding.json").read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Different experiment in {run}")
        selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        if (selection["experiment"] != expected or metrics["version"] != VERSION or
                metrics["target"] != target or
                metrics["n_test"] != manifest["source_report"][target]["gold_documents"] or
                metrics["selection_sha256"] != base.digest(run / "selection.json") or
                metrics["predictions_sha256"] != base.digest(run / "predictions.jsonl") or
                selection["curve_sha256"] != base.digest(run / "checkpoint_validation.csv") or
                historical.adapter_hashes(run / "selected_adapter") != selection["weights"]):
            raise ValueError(f"Incomplete or incompatible run: {target}")
        rows.append({"target": target, "n_test": metrics["n_test"],
                     "candidate": metrics["candidate"], "step": metrics["step"],
                     "threshold": metrics["threshold"], **metrics["metrics"],
                     **metrics["counts"]})
    macro = {key: sum(row[key] for row in rows) / len(rows) for key in METRICS}
    rows.append({"target": "MACRO_6_CORPUS", "n_test": sum(r["n_test"] for r in rows),
                 "candidate": "by_target_source_validation", "step": "by_target",
                 "threshold": "by_target_source_validation", **macro,
                 **{key: sum(r.get(key, 0) for r in rows)
                    for key in elementary.COUNT_FIELDS}})
    base.write_csv(root / "summary.csv", rows,
                   ("target", "n_test", "candidate", "step", "threshold",
                    *METRICS, *elementary.COUNT_FIELDS))
    checkpoint.write_json_atomic(root / "summary.json",
                                 {"version": VERSION, "method": METHOD,
                                  "experiment": expected, "seed": base.SEED,
                                  "results": rows, "completed_at_utc": base.stamp()})
    print(json.dumps({"summary": str(root / "summary.csv"), "macro": macro}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("check", None), ("smoke", smoke), ("fit", fit),
                           ("summarize", summarize)):
        command = commands.add_parser(name)
        command.add_argument("--bundle", type=Path, default=elementary.BASE_BUNDLE)
        command.add_argument("--prepared", type=Path, default=elementary.PREPARED)
        command.add_argument("--run-root", type=Path, required=True)
        if name in {"smoke", "fit"}:
            command.add_argument("--wtpsplit-repo", type=Path, required=True)
        if name == "fit":
            command.add_argument("--task-index", type=int, required=True)
        command.set_defaults(func=function)
    args = parser.parse_args()
    if args.command == "check":
        _, _, _, expected = checked_inputs(args)
        print(json.dumps({"ready": True, "experiment": expected}))
    else:
        args.func(args)


if __name__ == "__main__":
    main()
