#!/usr/bin/env python3
"""Six new, shared-test LODO LoRA fits for the final segmentation campaign."""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

import sat_indomain_v3 as single
import sat_lodo_checkpoint_select as checkpoint
import sat_lodo_gold_v3 as sat
import rq1_final as campaign

SAVE_STEPS = 1000
MIN_STEPS = 2
POLL_SECONDS = 5
ADAPTER_FILES = ("adapter_config.json", "head_config.json",
                 "pytorch_adapter.bin", "pytorch_model_head.bin")


def signature(bundle: Path) -> dict:
    return {"bundle_manifest_sha256": sat.digest(bundle / "manifest.json"),
            "script_sha256": sat.digest(Path(__file__)),
            "checkpoint_helper_sha256": sat.digest(Path(checkpoint.__file__)),
            "training_template_sha256": sat.digest(sat.legacy.TEMPLATE)}


def bind(run: Path, bundle: Path) -> None:
    path = run / "binding.json"
    expected = signature(bundle)
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Run belongs to another bundle or code: {run}")
    elif run.exists() and any(run.iterdir()):
        raise ValueError(f"Unbound run directory: {run}")
    else:
        run.mkdir(parents=True, exist_ok=True)
        campaign.json_file(path, expected)


def adapter_hashes(path: Path) -> dict:
    if not checkpoint.complete_adapter(path):
        raise FileNotFoundError(f"Incomplete adapter: {path}")
    return {name: sat.digest(path / name) for name in
            ("pytorch_adapter.bin", "pytorch_model_head.bin")}


def remove_checkpoint(trainer: Path, folder: Path) -> int:
    """Delete only a numbered checkpoint directly inside this run's trainer."""
    trainer = trainer.resolve()
    folder = folder.resolve()
    if (folder.parent != trainer or not folder.name.startswith("checkpoint-") or
            not folder.name.removeprefix("checkpoint-").isdigit()):
        raise ValueError(f"Refusing to remove a path outside the trainer: {folder}")
    size = sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())
    shutil.rmtree(folder)
    return size


def completed_checkpoint(folder: Path) -> dict | None:
    required = ("pytorch_model.bin", "optimizer.pt", "scheduler.pt", "trainer_state.json")
    if not all((folder / name).is_file() and (folder / name).stat().st_size > 0
               for name in required):
        return None
    try:
        state = json.loads((folder / "trainer_state.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None  # A save may still be in progress.
    step = int(folder.name.removeprefix("checkpoint-"))
    if int(state["global_step"]) != step:
        return None
    return {"step": step, "epoch": float(state["epoch"])}


def harvest_checkpoints(run: Path, trainer: Path, template: Path) -> None:
    """Export every finished checkpoint, retaining only the newest full state."""
    if not trainer.resolve().is_relative_to(run.resolve()):
        raise ValueError("Trainer directory escapes this run")
    folders = sorted((path for path in trainer.glob("checkpoint-*")
                      if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()),
                     key=lambda path: int(path.name.removeprefix("checkpoint-")))
    complete = []
    for folder in folders:
        snapshot = completed_checkpoint(folder)
        if snapshot is None:
            continue
        name = folder.name
        adapter = run / "checkpoint_adapters" / name
        metadata_path = adapter / "candidate.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (metadata["step"] != snapshot["step"] or
                    metadata["epoch"] != snapshot["epoch"] or
                    metadata["weights"] != adapter_hashes(adapter)):
                raise ValueError(f"Exported checkpoint changed: {adapter}")
        else:
            fingerprint = checkpoint.export_checkpoint_adapter(folder, adapter, template)
            metadata = {**snapshot, "checkpoint_sha256": fingerprint,
                        "weights": adapter_hashes(adapter)}
            campaign.json_file(metadata_path, metadata)
            print(f"Exported {name} to compact LoRA adapter", flush=True)
        complete.append((folder, metadata))
    if complete:
        for folder, metadata in complete[:-1]:
            marker = json.loads((run / "checkpoint_adapters" / folder.name /
                                 "checkpoint_source.json").read_text(encoding="utf-8"))
            if (marker["checkpoint_sha256"] != metadata["checkpoint_sha256"] or
                    sat.digest(folder / "pytorch_model.bin") != metadata["checkpoint_sha256"]):
                raise ValueError(f"Cannot safely prune changed checkpoint: {folder}")
            released = remove_checkpoint(trainer, folder)
            print(f"Pruned {folder.name}: {released / (1024 ** 3):.2f} GiB released",
                  flush=True)


def run_trainer_with_harvest(repo: Path, config: dict, config_path: Path,
                             run: Path, template: Path) -> None:
    """Monitor the unchanged upstream trainer while it saves full checkpoints."""
    campaign.json_file(config_path, config)
    script = (checkpoint.RESUME_SCRIPT if "resume_from_checkpoint" in config
              else sat.legacy.official_trainer(repo))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["WANDB_DISABLED"] = "true"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    process = subprocess.Popen([sys.executable, "-u", str(script), str(config_path)],
                               cwd=repo, env=environment)
    try:
        while process.poll() is None:
            harvest_checkpoints(run, run / "trainer", template)
            time.sleep(POLL_SECONDS)
        harvest_checkpoints(run, run / "trainer", template)
    except BaseException:
        if process.poll() is None:
            process.terminate()
        process.wait()
        raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, process.args)


def candidates_from_exports(run: Path, trainer: Path) -> list[dict]:
    state = json.loads((trainer / "trainer_state.json").read_text(encoding="utf-8"))
    final_step = int(state["global_step"])
    if final_step < 1:
        raise ValueError("Trainer completed without an optimizer step")
    result = []
    for step in range(SAVE_STEPS, final_step + 1, SAVE_STEPS):
        adapter = run / "checkpoint_adapters" / f"checkpoint-{step}"
        metadata = json.loads((adapter / "candidate.json").read_text(encoding="utf-8"))
        if metadata["step"] != step or metadata["weights"] != adapter_hashes(adapter):
            raise ValueError(f"Incomplete checkpoint candidate: {adapter}")
        if step != final_step:
            result.append({"candidate": f"checkpoint-{step}", "step": step,
                           "epoch": metadata["epoch"], "adapter": adapter})
    result.append({"candidate": "final", "step": final_step,
                   "epoch": float(state["epoch"]),
                   "adapter": sat.legacy.adapter_folder(trainer)})
    return result


def smoke(args: argparse.Namespace) -> None:
    import torch

    bundle, root = args.bundle.resolve(), args.run_root.resolve()
    manifest = campaign.check_bundle(bundle)
    marker = root / "smoke_ok.json"
    if marker.is_file():
        if json.loads(marker.read_text(encoding="utf-8")) != signature(bundle):
            raise ValueError("Smoke marker belongs to a different campaign")
        smoke_trainer = root / "smoke" / "trainer"
        if not smoke_trainer.resolve().is_relative_to((root / "smoke").resolve()):
            raise ValueError("Smoke trainer escapes its run")
        for folder in smoke_trainer.glob("checkpoint-*"):
            if folder.is_dir() and folder.name.removeprefix("checkpoint-").isdigit():
                remove_checkpoint(smoke_trainer, folder)
        print(f"Smoke already passed: {marker}", flush=True)
        return
    target = manifest["fold_order"][0]
    run = root / "smoke"
    bind(run, bundle)
    data = torch.load(bundle / "folds" / f"target_{target}" / "training.pth",
                      map_location="cpu", weights_only=False)
    original = data["fr"]["sentence"][sat.DATASET_NAME]
    sample = {"fr": {"sentence": {sat.DATASET_NAME: {
        "meta": {"train_data": original["meta"]["train_data"][:8]},
        "data": original["data"][:4]}}}}
    sample_path = run / "training.pth"
    torch.save(sample, sample_path)
    model_dir, tokenizer_dir = sat.pinned_backbone(bundle)
    config = sat.trainer_config(bundle, target, run / "trainer", model_dir, tokenizer_dir)
    config.update({"text_path": str(sample_path), "num_train_epochs": 1,
                   "max_steps": MIN_STEPS, "save_steps": 1, "save_total_limit": None,
                   "per_device_train_batch_size": 2, "per_device_eval_batch_size": 2,
                   "gradient_accumulation_steps": 1})
    repo = args.wtpsplit_repo.resolve()
    sat.legacy.git_revision(repo)
    if not checkpoint.complete_adapter(sat.legacy.adapter_folder(run / "trainer")):
        checkpoint.run_upstream(repo, config, run / "config.json")
    final = sat.legacy.adapter_folder(run / "trainer")
    adapter_hashes(final)
    intermediate = run / "trainer" / "checkpoint-1"
    exported = run / "checkpoint_adapter"
    checkpoint.export_checkpoint_adapter(intermediate, exported, final)
    adapter_hashes(exported)
    narrative = sat.read_jsonl(bundle / "folds" / f"target_{target}" / "val.jsonl")[0]["raw_text"]
    for adapter in (exported, final):
        model = sat.load_pinned_sat(model_dir, tokenizer_dir, adapter)
        sat.legacy.pieces_to_spans(narrative, model.split(narrative, threshold=sat.DEFAULT))
        del model
    root.mkdir(parents=True, exist_ok=True)
    campaign.json_file(marker, signature(bundle))
    for folder in (run / "trainer").glob("checkpoint-*"):
        if folder.is_dir() and folder.name.removeprefix("checkpoint-").isdigit():
            remove_checkpoint(run / "trainer", folder)
    print(f"Smoke passed: {marker}", flush=True)


def persist_selected_adapter(source: Path, destination: Path, expected: dict) -> None:
    if adapter_hashes(source) != expected:
        raise ValueError(f"Selected source weights changed: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ADAPTER_FILES:
        target = destination / name
        if target.is_file():
            if sat.digest(target) != sat.digest(source / name):
                raise ValueError(f"Saved selected adapter changed: {target}")
        else:
            shutil.copy2(source / name, target)
    if adapter_hashes(destination) != expected:
        raise ValueError("Saved selected adapter differs from selected checkpoint")


def cleanup_full_states(run: Path, selection: dict) -> None:
    """After complete scoring, retain only the selected adapter and audit records."""
    trainer = run / "trainer"
    selected = run / "selected_adapter"
    if (not trainer.resolve().is_relative_to(run.resolve()) or
            selected.resolve().parent != run.resolve()):
        raise ValueError("Cleanup path escapes this run")
    if adapter_hashes(selected) != selection["M3_weights"]:
        raise ValueError("Cannot clean up before selected weights are safely saved")
    metrics_path = run / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (metrics["selection_sha256"] != sat.digest(run / "selection.json") or
            any(metrics["prediction_sha256"][label] != sat.digest(
                run / "predictions" / f"{label}.jsonl") for label in ("M1", "M2", "M3"))):
        raise ValueError("Cannot clean up before all test outputs are verified")
    released = 0
    for folder in trainer.glob("checkpoint-*"):
        if folder.is_dir() and folder.name.removeprefix("checkpoint-").isdigit():
            released += remove_checkpoint(trainer, folder)
    adapters = run / "checkpoint_adapters"
    if adapters.is_dir():
        if adapters.resolve().parent != run.resolve():
            raise ValueError("Refusing to remove adapters outside this run")
        released += sum(path.stat().st_size for path in adapters.rglob("*") if path.is_file())
        shutil.rmtree(adapters)
    final = sat.legacy.adapter_folder(trainer)
    if final.is_dir():
        if not final.resolve().is_relative_to(trainer.resolve()):
            raise ValueError("Refusing to remove final adapter outside this trainer")
        released += sum(path.stat().st_size for path in final.rglob("*") if path.is_file())
        shutil.rmtree(final)
    campaign.json_file(run / "storage_cleanup.json", {
        "selection_sha256": sat.digest(run / "selection.json"),
        "metrics_sha256": sat.digest(metrics_path),
        "selected_weights": selection["M3_weights"],
        "released_bytes_this_pass": released,
        "full_checkpoints_remaining": 0})
    print(f"Retained selected adapter only; released {released / (1024 ** 3):.2f} GiB",
          flush=True)


def fit(args: argparse.Namespace) -> None:
    import torch

    bundle, root = args.bundle.resolve(), args.run_root.resolve()
    manifest = campaign.check_bundle(bundle)
    if not 0 <= args.task_index < 6:
        raise ValueError("Task index must be 0..5")
    if json.loads((root / "smoke_ok.json").read_text(encoding="utf-8")) != signature(bundle):
        raise ValueError("Smoke not completed for this campaign")
    target = manifest["fold_order"][args.task_index]
    run = root / f"target_{target}"
    bind(run, bundle)
    metrics_path = run / "metrics.json"
    if metrics_path.is_file():
        selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
        cleanup_full_states(run, selection)
        print(f"Already completed {target}; selected adapter verified", flush=True)
        return
    model_dir, tokenizer_dir = sat.pinned_backbone(bundle)
    repo = args.wtpsplit_repo.resolve()
    revision = sat.legacy.git_revision(repo)
    runtime = sat.legacy.gpu_runtime(repo)
    trainer = run / "trainer"
    config = sat.trainer_config(bundle, target, trainer, model_dir, tokenizer_dir)
    config.update({"save_strategy": "steps", "save_steps": SAVE_STEPS,
                   "save_total_limit": None, "load_best_model_at_end": False})
    final = sat.legacy.adapter_folder(trainer)
    if not checkpoint.complete_adapter(final):
        latest = checkpoint.latest_resumable_checkpoint(trainer)
        if latest:
            config["resume_from_checkpoint"] = str(latest)
            print(f"Resuming {target} from {latest}", flush=True)
        template = sat.legacy.adapter_folder(root / "smoke" / "trainer")
        adapter_hashes(template)
        run_trainer_with_harvest(repo, config, run / "config.json", run, template)
    adapter_hashes(final)
    harvest_checkpoints(run, trainer, sat.legacy.adapter_folder(root / "smoke" / "trainer"))
    choices = candidates_from_exports(run, trainer)
    state = trainer / "trainer_state.json"
    sat.legacy.export_loss_history(state, run / "loss_history.csv")
    campaign.json_file(run / "training_done.json", {
        "target": target, "seed": config["seed"], "bundle": signature(bundle),
        "config_sha256": sat.digest(run / "config.json"),
        "wtpsplit_revision": revision, "runtime": runtime,
        "checkpoint_steps": [c["step"] for c in choices],
        "model_revision": model_dir.name, "tokenizer_revision": tokenizer_dir.name})
    val = sat.read_jsonl(bundle / "folds" / f"target_{target}" / "val.jsonl")
    curves = []
    adapter_paths = {}
    for candidate in choices:
        name = candidate["candidate"]
        adapter = candidate["adapter"]
        hashes = adapter_hashes(adapter)
        adapter_paths[name] = adapter
        cache = run / "validation_cache" / f"{name}.sqlite3"
        binding = cache.with_suffix(".binding.json")
        expected = {"bundle": signature(bundle), "candidate": name, "weights": hashes}
        if binding.is_file():
            if json.loads(binding.read_text(encoding="utf-8")) != expected:
                raise ValueError(f"Validation cache has different weights: {cache}")
        elif cache.exists():
            raise ValueError(f"Unbound cache: {cache}")
        else:
            campaign.json_file(binding, expected)
        with closing(sat._probability_connection(cache, create=True)) as db:
            if db.execute("SELECT COUNT(*) FROM probabilities").fetchone()[0] < len(val):
                model = sat.load_pinned_sat(model_dir, tokenizer_dir, adapter)
                single.infer_fixed_batches(model, db, val, f"Validation {target}/{name}")
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if db.execute("SELECT COUNT(*) FROM probabilities").fetchone()[0] != len(val):
                raise ValueError("Incomplete validation")
            curves.extend(checkpoint.validation_curve(
                val, db, name, candidate["step"], candidate["epoch"]))
    sat.write_csv(run / "checkpoint_validation.csv", curves, checkpoint.CURVE_FIELDS)
    chosen = checkpoint.best_row(curves)
    with closing(sat._probability_connection(bundle / "base_probabilities.sqlite3")) as db:
        base_curve = checkpoint.validation_curve(val, db, "base", -1, -1.0)
    sat.write_csv(run / "base_validation.csv", base_curve, checkpoint.CURVE_FIELDS)
    calibrated = checkpoint.best_row(base_curve)
    selection = {"target": target, "criterion": "macro of five source micro exact-span F1s",
                 "n_validation": len(val), "M2_threshold": calibrated["threshold"],
                 "M2_validation_f1": calibrated["source_macro_span_f1"],
                 "M3_candidate": chosen["candidate"], "M3_step": chosen["step"],
                 "M3_threshold": chosen["threshold"],
                 "M3_validation_f1": chosen["source_macro_span_f1"],
                 "M3_weights": adapter_hashes(adapter_paths[chosen["candidate"]]),
                 "bundle": signature(bundle),
                 "checkpoint_validation_sha256": sat.digest(run / "checkpoint_validation.csv"),
                 "base_validation_sha256": sat.digest(run / "base_validation.csv")}
    selected_path = run / "selection.json"
    if selected_path.exists():
        if json.loads(selected_path.read_text(encoding="utf-8")) != selection:
            raise ValueError("Selection changed on resume")
    else:
        campaign.json_file(selected_path, selection)
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)
    test = sat.read_jsonl(bundle / "folds" / f"target_{target}" / "test.jsonl")
    selected_adapter = run / "selected_adapter"
    persist_selected_adapter(adapter_paths[chosen["candidate"]], selected_adapter,
                             selection["M3_weights"])
    cache = run / "test_probabilities.sqlite3"
    binding = cache.with_suffix(".binding.json")
    expected = {"bundle": signature(bundle), "weights": selection["M3_weights"],
                "selected_candidate": chosen["candidate"]}
    if binding.exists():
        if json.loads(binding.read_text(encoding="utf-8")) != expected:
            raise ValueError("Test cache uses another adapter")
    elif cache.exists():
        raise ValueError("Unbound test cache")
    else:
        campaign.json_file(binding, expected)
    with closing(sat._probability_connection(cache, create=True)) as db:
        if db.execute("SELECT COUNT(*) FROM probabilities").fetchone()[0] < len(test):
            model = sat.load_pinned_sat(model_dir, tokenizer_dir, selected_adapter)
            single.infer_fixed_batches(model, db, test, f"Test {target}")
            sat.verify_native(model, db, test, selection["M3_threshold"])
            del model
        if db.execute("SELECT COUNT(*) FROM probabilities").fetchone()[0] != len(test):
            raise ValueError("Incomplete target test cache")
        with closing(sat._probability_connection(bundle / "base_probabilities.sqlite3")) as baseline:
            rows = []
            pred_dir = run / "predictions"
            pred_dir.mkdir(exist_ok=True)
            streams = {label: (pred_dir / f"{label}.jsonl.tmp").open("w", encoding="utf-8")
                       for label in ("M1", "M2", "M3")}
            try:
                for row in test:
                    raw = row["raw_text"]
                    base_probs = sat.get_probs(baseline, row)
                    adapted_probs = sat.get_probs(db, row)
                    for label, threshold, probs in (
                        ("M1", sat.DEFAULT, base_probs),
                        ("M2", selection["M2_threshold"], base_probs),
                        ("M3", selection["M3_threshold"], adapted_probs),
                    ):
                        spans = sat.legacy.spans_from_probabilities(raw, probs, threshold)
                        rows.append({"target": target, "document_id": row["document_id"],
                                     "model": label, "threshold": threshold,
                                     **sat.score(row, spans, details=True)})
                        streams[label].write(json.dumps({"corpus": target,
                            "document_id": row["document_id"], "threshold": threshold,
                            "segments": [{"start": a, "end": b, "text": raw[a:b]}
                                         for a, b in spans]}, ensure_ascii=False) + "\n")
            finally:
                for stream in streams.values():
                    stream.close()
            for label in streams:
                (pred_dir / f"{label}.jsonl.tmp").replace(pred_dir / f"{label}.jsonl")
    fields = ("target", "document_id", "model", "threshold", *sat.SCORE_FIELDS,
              *(f"role_{role}_{kind}" for role in sat.ROLES for kind in ("tp", "ref")),
              "fusion", "fragmented", "gold_units", "pred_segments", "supported_pred_segments")
    sat.write_csv(run / "document_metrics.csv", rows, fields)
    metrics = {label: sat.aggregate_metrics(rows, label, target)
               for label in ("M1", "M2", "M3")}
    campaign.json_file(metrics_path, {
        "target": target, "reference": sat.REFERENCE, "bundle": signature(bundle),
        "selection_sha256": sat.digest(selected_path), "selection": selection,
        "test": metrics, "n_test": len(test),
        "prediction_sha256": {label: sat.digest(run / "predictions" / f"{label}.jsonl")
                              for label in ("M1", "M2", "M3")}})
    cleanup_full_states(run, selection)
    print(f"Completed {target}: M3 exact F1={metrics['M3']['span_f1']:.5f}", flush=True)


def summarize(args: argparse.Namespace) -> None:
    bundle, root = args.bundle.resolve(), args.run_root.resolve()
    manifest = campaign.check_bundle(bundle)
    rows = []
    for target in sat.CORPORA:
        run = root / f"target_{target}"
        result = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
        cleanup = json.loads((run / "storage_cleanup.json").read_text(encoding="utf-8"))
        if (result["target"] != target or result["bundle"] != signature(bundle)
                or result["selection_sha256"] != sat.digest(run / "selection.json")
                or result["n_test"] != campaign.EXPECTED_COUNTS[target][1]
                or result["selection"] != selection
                or cleanup["metrics_sha256"] != sat.digest(run / "metrics.json")
                or cleanup["selected_weights"] != selection["M3_weights"]
                or cleanup["full_checkpoints_remaining"] != 0):
            raise ValueError(f"Incomplete or mismatched rotation: {target}")
        for label in ("M1", "M2", "M3"):
            path = run / "predictions" / f"{label}.jsonl"
            if result["prediction_sha256"][label] != sat.digest(path):
                raise ValueError(f"Prediction file changed: {path}")
        rows.append({"target": target, "n_test": result["n_test"],
                     "M2_threshold": selection["M2_threshold"],
                     "M3_threshold": selection["M3_threshold"],
                     "M3_checkpoint": selection["M3_candidate"],
                     **{f"{label}_exact_f1": result["test"][label]["span_f1"]
                        for label in ("M1", "M2", "M3")}})
    sat.write_csv(root / "cluster_summary.csv", rows, tuple(rows[0]))
    completion = {
        "campaign": campaign.NAME, "gold_snapshot_sha256": manifest["gold_snapshot_sha256"],
        "common_split_sha256": manifest["common_split_sha256"],
        "bundle_manifest_sha256": sat.digest(bundle / "manifest.json"),
        "model": sat.MODEL, "one_seed": True, "results": rows}
    campaign.json_file(root / "cluster_complete.json", completion)
    transfer = root / "report_transfer"
    transfer.mkdir(exist_ok=True)
    files = {}
    sources = [(root / "cluster_complete.json", transfer / "cluster_complete.json")]
    for target in sat.CORPORA:
        run = root / f"target_{target}"
        selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
        adapter = run / "selected_adapter"
        if adapter_hashes(adapter) != selection["M3_weights"]:
            raise ValueError(f"Selected weights changed before transfer: {target}")
        for name in ("selection.json", "metrics.json"):
            sources.append((run / name, transfer / f"target_{target}" / name))
        for label in ("M1", "M2", "M3"):
            sources.append((run / "predictions" / f"{label}.jsonl",
                            transfer / f"target_{target}" / "predictions" / f"{label}.jsonl"))
        for name in ("adapter_config.json", "head_config.json", "pytorch_adapter.bin",
                     "pytorch_model_head.bin"):
            sources.append((adapter / name,
                            transfer / f"target_{target}" / "selected_adapter" / name))
    for source, dest in sources:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_file():
            if sat.digest(dest) != sat.digest(source):
                raise ValueError(f"Existing transfer member changed: {dest}")
        else:
            shutil.copy2(source, dest)
        files[dest.relative_to(transfer).as_posix()] = sat.digest(dest)
    campaign.json_file(transfer / "transfer_manifest.json", {
        "campaign": campaign.NAME, "common_split_sha256": manifest["common_split_sha256"],
        "gold_snapshot_sha256": manifest["gold_snapshot_sha256"], "files": files})
    print(json.dumps({"complete": True, "results": rows}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("smoke", smoke), ("fit", fit), ("summarize", summarize)):
        command = commands.add_parser(name)
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--run-root", type=Path, required=True)
        if name != "summarize":
            command.add_argument("--wtpsplit-repo", type=Path, required=True)
        if name == "fit":
            command.add_argument("--task-index", type=int, required=True)
        command.set_defaults(func=function)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
