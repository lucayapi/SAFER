#!/usr/bin/env python3
"""SaT+LoRA LODO with checkpoint/threshold selection on source validation only."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

import torch

import sat_lodo_checkpoint_select as checkpoint
import sat_narrative_only_lodo as m3
import sat_narrative_only_runtime as runtime

SAVE_STEPS = 500
POLL_SECONDS = 5
ADAPTER_FILES = ("adapter_config.json", "head_config.json",
                 "pytorch_adapter.bin", "pytorch_model_head.bin")
METRIC = "source_macro_edge_normalized_exact_f1"
TEST_METRIC = "edge_normalized_exact_f1"
CURVE_FIELDS = ("candidate", "step", "epoch", "threshold", METRIC,
                *(f"f1_{corpus}" for corpus in runtime.CORPORA))


def digest(path: Path) -> str:
    return runtime.digest(path)


def script_digest() -> str:
    h = hashlib.sha256()
    for path in (Path(__file__), Path(checkpoint.__file__)):
        h.update(path.name.encode("utf-8"))
        h.update(path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8"))
    return h.hexdigest()


def campaign_binding(prepared: Path) -> dict:
    return {"m3_binding": m3.binding(prepared), "script_sha256": script_digest(),
            "checkpoint_interval_steps": SAVE_STEPS,
            "selection_metric": METRIC}


def write_json(path: Path, value: object) -> None:
    runtime.write_json(path, value)


def adapter_hashes(path: Path) -> dict:
    if not checkpoint.complete_adapter(path):
        raise FileNotFoundError(f"Incomplete LoRA adapter: {path}")
    return {name: digest(path / name) for name in
            ("pytorch_adapter.bin", "pytorch_model_head.bin")}


def safe_remove_checkpoint(trainer: Path, folder: Path) -> int:
    trainer, folder = trainer.resolve(), folder.resolve()
    if (folder.parent != trainer or not folder.name.startswith("checkpoint-") or
            not folder.name.removeprefix("checkpoint-").isdigit()):
        raise ValueError(f"Refusing to remove unexpected checkpoint path: {folder}")
    size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
    shutil.rmtree(folder)
    return size


def complete_checkpoint(folder: Path) -> dict | None:
    names = ("pytorch_model.bin", "optimizer.pt", "scheduler.pt", "trainer_state.json")
    if not all((folder / n).is_file() and (folder / n).stat().st_size for n in names):
        return None
    try:
        state = json.loads((folder / "trainer_state.json").read_text(encoding="utf-8"))
        step = int(folder.name.removeprefix("checkpoint-"))
        if int(state["global_step"]) != step:
            return None
        return {"step": step, "epoch": float(state["epoch"])}
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def harvest(run: Path, trainer: Path, template: Path) -> None:
    """Export all finished 500-step LoRA candidates and retain one full state."""
    if not trainer.resolve().is_relative_to(run.resolve()):
        raise ValueError("Trainer directory escapes its run directory")
    folders = sorted((p for p in trainer.glob("checkpoint-*")
                      if p.is_dir() and p.name.removeprefix("checkpoint-").isdigit()),
                     key=lambda p: int(p.name.removeprefix("checkpoint-")))
    finished = []
    for folder in folders:
        info = complete_checkpoint(folder)
        if info is None:
            continue
        step = info["step"]
        if step % SAVE_STEPS:
            continue
        adapter = run / "checkpoint_adapters" / folder.name
        metadata_path = adapter / "candidate.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (metadata.get("step") != step or metadata.get("epoch") != info["epoch"] or
                    metadata.get("weights") != adapter_hashes(adapter) or
                    metadata.get("checkpoint_sha256") != digest(folder / "pytorch_model.bin")):
                raise ValueError(f"Saved candidate no longer matches {folder}")
        else:
            fingerprint = checkpoint.export_checkpoint_adapter(folder, adapter, template)
            metadata = {**info, "checkpoint_sha256": fingerprint,
                        "weights": adapter_hashes(adapter)}
            write_json(metadata_path, metadata)
            print(f"Exported LoRA candidate at step {step}", flush=True)
        finished.append((folder, metadata))
    # Keep the newest complete full checkpoint so interrupted training can resume.
    for folder, metadata in finished[:-1]:
        marker = json.loads((run / "checkpoint_adapters" / folder.name /
                             "checkpoint_source.json").read_text(encoding="utf-8"))
        if (marker.get("checkpoint_sha256") != metadata["checkpoint_sha256"] or
                digest(folder / "pytorch_model.bin") != metadata["checkpoint_sha256"]):
            raise ValueError(f"Refusing to prune unverified checkpoint {folder}")
        freed = safe_remove_checkpoint(trainer, folder)
        print(f"Removed {folder.name}; freed {freed / (1024 ** 3):.2f} GiB", flush=True)


def run_trainer(repo: Path, config: dict, run: Path, template: Path) -> None:
    config_path = run / "config.json"
    write_json(config_path, config)
    command = (checkpoint.RESUME_SCRIPT if "resume_from_checkpoint" in config
               else runtime.legacy.official_trainer(repo))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["WANDB_DISABLED"] = "true"
    env["TOKENIZERS_PARALLELISM"] = "false"
    proc = subprocess.Popen([sys.executable, "-u", str(command), str(config_path)],
                            cwd=repo, env=env)
    try:
        while proc.poll() is None:
            harvest(run, run / "trainer", template)
            time.sleep(POLL_SECONDS)
        harvest(run, run / "trainer", template)
    except BaseException:
        if proc.poll() is None:
            proc.terminate()
        proc.wait()
        raise
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, proc.args)


def candidate_list(run: Path, trainer: Path) -> list[dict]:
    state = json.loads((trainer / "trainer_state.json").read_text(encoding="utf-8"))
    final_step = int(state["global_step"])
    if final_step < 1:
        raise ValueError("Training completed without an optimizer step")
    result = []
    candidate_root = run / "checkpoint_adapters"
    for folder in sorted(candidate_root.glob("checkpoint-*"),
                         key=lambda p: int(p.name.removeprefix("checkpoint-"))):
        if not folder.is_dir() or not folder.name.removeprefix("checkpoint-").isdigit():
            continue
        metadata = json.loads((folder / "candidate.json").read_text(encoding="utf-8"))
        step = int(folder.name.removeprefix("checkpoint-"))
        if (step % SAVE_STEPS or metadata.get("step") != step or
                metadata.get("weights") != adapter_hashes(folder)):
            raise ValueError(f"Invalid exported candidate: {folder}")
        result.append({"candidate": folder.name, "step": step,
                       "epoch": metadata["epoch"], "adapter": folder})
    final_adapter = runtime.legacy.adapter_folder(trainer)
    result.append({"candidate": "final", "step": final_step,
                   "epoch": float(state["epoch"]), "adapter": final_adapter})
    if not result:
        raise ValueError("No checkpoint candidates found")
    return result


def validation_curve(val: list[dict], db, candidate: dict) -> list[dict]:
    sources = sorted({row["corpus"] for row in val})
    if len(sources) != 5:
        raise ValueError("Checkpoint selection requires the five source validation corpora")
    totals = collections.defaultdict(collections.Counter)
    for row in val:
        if not row["raw_text"]:
            continue
        probs = runtime.get_probs(db, row)
        for threshold in runtime.THRESHOLDS:
            spans = runtime.legacy.spans_from_probabilities(row["raw_text"], probs, threshold)
            totals[row["corpus"], threshold].update(m3.score(row, spans))
    curve = []
    for threshold in runtime.THRESHOLDS:
        per_source = {source: m3.prf(totals[source, threshold])[2] for source in sources}
        curve.append({"candidate": candidate["candidate"], "step": candidate["step"],
                      "epoch": candidate["epoch"], "threshold": threshold,
                      METRIC: sum(per_source.values()) / len(per_source),
                      **{f"f1_{source}": per_source[source] for source in runtime.CORPORA}})
    return curve


def select_candidate(curve: list[dict]) -> dict:
    if not curve:
        raise ValueError("No source-validation checkpoint scores to select from")
    return min(curve, key=lambda row: (-row[METRIC],
        abs(float(row["threshold"]) - runtime.DEFAULT), int(row["step"]),
        float(row["threshold"])))


def smoke(args: argparse.Namespace) -> None:
    prepared, root = args.prepared.resolve(), args.run_root.resolve()
    m3.check_prepared(prepared)
    expected = campaign_binding(prepared)
    marker = root / "checkpoint_smoke_ok.json"
    if marker.is_file():
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ValueError("Existing smoke marker belongs to another run")
        print(f"Checkpoint-selection smoke already passed: {marker}", flush=True)
        return
    repo = args.wtpsplit_repo.resolve()
    runtime.legacy.gpu_runtime(repo)
    model_dir, tokenizer_dir = m3.pinned_backbone(prepared)
    target = runtime.CORPORA[0]
    fold = prepared / "folds" / f"target_{target}"
    data = torch.load(fold / "training.pth", map_location="cpu", weights_only=False)
    original = data["fr"]["sentence"][runtime.DATASET_NAME]
    work = root / "checkpoint_smoke"
    work.mkdir(parents=True, exist_ok=True)
    sample = work / "training.pth"
    torch.save({"fr": {"sentence": {runtime.DATASET_NAME: {
        "meta": {"train_data": original["meta"]["train_data"][:8]},
        "data": original["data"][:4]}}}}, sample)
    config = m3.trainer_config(target, work / "trainer", model_dir, tokenizer_dir, prepared)
    config.update({"text_path": str(sample), "num_train_epochs": 1, "max_steps": 1,
                   "save_strategy": "steps", "save_steps": 1, "save_total_limit": None,
                   "per_device_train_batch_size": 2, "per_device_eval_batch_size": 2,
                   "gradient_accumulation_steps": 1})
    runtime.legacy.run_trainer(repo, config, work / "config.json")
    trainer = work / "trainer"
    final = runtime.legacy.adapter_folder(trainer)
    intermediate = trainer / "checkpoint-1"
    exported = work / "checkpoint_adapter"
    checkpoint.export_checkpoint_adapter(intermediate, exported, final)
    adapter_hashes(exported)
    row = next(r for r in runtime.read_jsonl(fold / "val.jsonl") if r["raw_text"])
    model = runtime.load_pinned_sat(model_dir, tokenizer_dir, exported)
    runtime.legacy.pieces_to_spans(row["raw_text"], model.split(row["raw_text"], threshold=runtime.DEFAULT))
    del model
    write_json(marker, expected)
    print(f"Checkpoint-selection smoke passed: {marker}", flush=True)


def fit(args: argparse.Namespace) -> None:
    prepared, root = args.prepared.resolve(), args.run_root.resolve()
    manifest = m3.check_prepared(prepared)
    smoke_marker = root / "checkpoint_smoke_ok.json"
    if not smoke_marker.is_file() or json.loads(smoke_marker.read_text(encoding="utf-8")) != campaign_binding(prepared):
        raise RuntimeError("Run the checkpoint-selection smoke first")
    if not 0 <= args.task_index < len(runtime.CORPORA):
        raise ValueError("task-index must be 0..5")
    target = manifest["fold_order"][args.task_index]
    run = root / f"target_{target}"
    run.mkdir(parents=True, exist_ok=True)
    binding_path = run / "binding.json"
    expected_m3_binding = m3.binding(prepared)
    if binding_path.exists() and json.loads(binding_path.read_text(encoding="utf-8")) != expected_m3_binding:
        raise ValueError(f"Run belongs to another prepared bundle: {run}")
    if not binding_path.exists():
        if any(run.iterdir()):
            raise ValueError(f"Refusing unbound nonempty run directory: {run}")
        write_json(binding_path, expected_m3_binding)
    metrics_path = run / "metrics.json"
    if metrics_path.is_file():
        selection_path = run / "selection.json"
        if not selection_path.is_file():
            raise ValueError(f"Completed metrics lack checkpoint selection: {selection_path}")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if not (run / "storage_cleanup.json").is_file():
            cleanup(run, selection)
        print(f"Already complete: {target}", flush=True)
        return

    repo = args.wtpsplit_repo.resolve()
    runtime.legacy.gpu_runtime(repo)
    model_dir, tokenizer_dir = m3.pinned_backbone(prepared)
    trainer = run / "trainer"
    config = m3.trainer_config(target, trainer, model_dir, tokenizer_dir, prepared)
    config.update({"save_strategy": "steps", "save_steps": SAVE_STEPS,
                   "save_total_limit": None, "load_best_model_at_end": False})
    final_adapter = runtime.legacy.adapter_folder(trainer)
    if not checkpoint.complete_adapter(final_adapter):
        resumable = checkpoint.latest_resumable_checkpoint(trainer)
        if resumable:
            config["resume_from_checkpoint"] = str(resumable)
        template = runtime.legacy.adapter_folder(root / "checkpoint_smoke" / "trainer")
        if not checkpoint.complete_adapter(template):
            raise FileNotFoundError(f"Smoke adapter template missing: {template}")
        run_trainer(repo, config, run, template)
    if not checkpoint.complete_adapter(final_adapter):
        raise RuntimeError(f"Training did not save the final adapter: {final_adapter}")
    harvest(run, trainer, runtime.legacy.adapter_folder(root / "checkpoint_smoke" / "trainer"))
    candidates = candidate_list(run, trainer)
    state_path = trainer / "trainer_state.json"
    if state_path.is_file():
        runtime.legacy.export_loss_history(state_path, run / "loss_history.csv")
    write_json(run / "training_done.json", {"target": target, "seed": runtime.SEED,
        "binding": expected_m3_binding, "checkpoint_steps": [c["step"] for c in candidates],
        "config_sha256": digest(run / "config.json"), "model_revision": model_dir.name,
        "tokenizer_revision": tokenizer_dir.name, "selection_protocol": campaign_binding(prepared)})

    val = runtime.read_jsonl(prepared / "folds" / f"target_{target}" / "val.jsonl")
    all_curve = []
    for candidate in candidates:
        adapter = candidate["adapter"]
        name = candidate["candidate"]
        cache = run / "validation_cache" / f"{name}.sqlite3"
        cache_binding = cache.with_suffix(".binding.json")
        expected_cache = {"candidate": name, "step": candidate["step"],
                          "adapter_sha256": adapter_hashes(adapter),
                          "m3_binding": expected_m3_binding}
        if cache_binding.exists():
            if json.loads(cache_binding.read_text(encoding="utf-8")) != expected_cache:
                raise ValueError(f"Validation cache binding mismatch: {cache}")
        elif cache.exists():
            raise ValueError(f"Refusing unbound validation cache: {cache}")
        else:
            write_json(cache_binding, expected_cache)
        with closing(runtime._probability_connection(cache, create=True)) as db:
            count = db.execute("SELECT COUNT(*) FROM probabilities").fetchone()[0]
            if count < len([r for r in val if r["raw_text"]]):
                model = runtime.load_pinned_sat(model_dir, tokenizer_dir, adapter)
                runtime.infer_adapted(model, db, [r for r in val if r["raw_text"]])
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            expected_count = len([r for r in val if r["raw_text"]])
            if db.execute("SELECT COUNT(*) FROM probabilities").fetchone()[0] != expected_count:
                raise ValueError(f"Incomplete source validation probabilities: {name}")
            curve = validation_curve(val, db, candidate)
        all_curve.extend(curve)
        runtime.write_csv(run / f"source_validation_{name}.csv", curve, CURVE_FIELDS)
    runtime.write_csv(run / "checkpoint_validation.csv", all_curve, CURVE_FIELDS)
    selected = select_candidate(all_curve)
    selection = {"target": target, "criterion": "unweighted macro of the five source-corpus micro edge-normalized exact-span F1s",
        "tie_break": "threshold nearest 0.25, earlier optimizer step, then smaller threshold",
        "candidate": selected["candidate"], "step": int(selected["step"]),
        "epoch": float(selected["epoch"]), "threshold": float(selected["threshold"]),
        METRIC: float(selected[METRIC]), "adapter_sha256": adapter_hashes(
            next(c["adapter"] for c in candidates if c["candidate"] == selected["candidate"])),
        "m3_binding": expected_m3_binding, "campaign_binding": campaign_binding(prepared),
        "checkpoint_validation_sha256": digest(run / "checkpoint_validation.csv")}
    selection_path = run / "selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text(encoding="utf-8")) != selection:
            raise ValueError("Checkpoint selection changed on resume")
    else:
        write_json(selection_path, selection)
    selected_adapter_path = next(c["adapter"] for c in candidates
                                 if c["candidate"] == selected["candidate"])
    selected_adapter = run / "selected_adapter"
    selected_hashes = adapter_hashes(selected_adapter_path)
    selected_adapter.mkdir(exist_ok=True)
    for name in ADAPTER_FILES:
        dest = selected_adapter / name
        if not dest.exists():
            shutil.copy2(selected_adapter_path / name, dest)
        elif digest(dest) != digest(selected_adapter_path / name):
            raise ValueError(f"Selected adapter changed: {dest}")
    if adapter_hashes(selected_adapter) != selected_hashes:
        raise ValueError("Persisted selected adapter failed hash verification")

    # Target test is loaded only after checkpoint and threshold are fixed.
    test = runtime.read_jsonl(prepared / "folds" / f"target_{target}" / "test.jsonl")
    test_cache = run / "selected_test_probabilities.sqlite3"
    test_binding = test_cache.with_suffix(".binding.json")
    expected_test_binding = {"adapter_sha256": selected_hashes,
                             "candidate": selected["candidate"], "m3_binding": expected_m3_binding}
    if test_binding.exists():
        if json.loads(test_binding.read_text(encoding="utf-8")) != expected_test_binding:
            raise ValueError("Target-test cache is bound to another selected checkpoint")
    elif test_cache.exists():
        raise ValueError("Refusing unbound target-test cache")
    else:
        write_json(test_binding, expected_test_binding)
    with closing(runtime._probability_connection(test_cache, create=True)) as db:
        nonempty = [r for r in test if r["raw_text"]]
        count = db.execute("SELECT COUNT(*) FROM probabilities").fetchone()[0]
        if count < len(nonempty):
            model = runtime.load_pinned_sat(model_dir, tokenizer_dir, selected_adapter)
            runtime.infer_adapted(model, db, nonempty)
            runtime.verify_native(model, db, nonempty, float(selected["threshold"]))
            del model
        if db.execute("SELECT COUNT(*) FROM probabilities").fetchone()[0] != len(nonempty):
            raise ValueError("Incomplete target-test predictions")
        total = collections.Counter()
        with (run / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in test:
                spans = (runtime.legacy.spans_from_probabilities(row["raw_text"],
                    runtime.get_probs(db, row), float(selected["threshold"]))
                    if row["raw_text"] else [])
                total.update(m3.score(row, spans))
                stream.write(json.dumps({"corpus": target, "document_id": row["document_id"],
                    "threshold": float(selected["threshold"]),
                    "segments": [{"start": a, "end": b} for a, b in spans]},
                    ensure_ascii=False) + "\n")
    p, r, f1 = m3.prf(total)
    metrics = {"version": m3.VERSION, "method": m3.METHOD, "target": target,
        "n_test": len(test), "seed": runtime.SEED, "threshold": float(selected["threshold"]),
        "normalization": "trim Unicode whitespace and punctuation at span edges only",
        "selection": selection, "binding": expected_m3_binding,
        "counts": {k: int(total[k]) for k in m3.FIELDS},
        "metrics": {"precision": p, "recall": r, TEST_METRIC: f1},
        "prediction_sha256": digest(run / "predictions.jsonl"),
        "selection_sha256": digest(run / "selection.json"), "completed_at_utc": runtime.stamp()}
    write_json(metrics_path, metrics)
    cleanup(run, selection)
    print(json.dumps({"target": target, "selected": selected["candidate"],
        "step": selected["step"], "threshold": selected["threshold"],
        "source_macro_f1": selected[METRIC], "test_f1": f1}, ensure_ascii=False, indent=2), flush=True)


def cleanup(run: Path, selection: dict) -> None:
    trainer, selected = run / "trainer", run / "selected_adapter"
    if not trainer.resolve().is_relative_to(run.resolve()) or selected.resolve().parent != run.resolve():
        raise ValueError("Cleanup path escapes the run")
    if adapter_hashes(selected) != selection["adapter_sha256"]:
        raise ValueError("Selected adapter failed verification before cleanup")
    metrics_path = run / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (metrics.get("prediction_sha256") != digest(run / "predictions.jsonl") or
            metrics.get("selection_sha256") != digest(run / "selection.json")):
        raise ValueError("Cannot clean up before test outputs are verified")
    released = 0
    for folder in trainer.glob("checkpoint-*"):
        if folder.is_dir() and folder.name.removeprefix("checkpoint-").isdigit():
            released += safe_remove_checkpoint(trainer, folder)
    adapters = run / "checkpoint_adapters"
    if adapters.is_dir():
        released += sum(p.stat().st_size for p in adapters.rglob("*") if p.is_file())
        shutil.rmtree(adapters)
    final_adapter = runtime.legacy.adapter_folder(trainer)
    if final_adapter.is_dir():
        released += sum(p.stat().st_size for p in final_adapter.rglob("*") if p.is_file())
        shutil.rmtree(final_adapter)
    validation_cache = run / "validation_cache"
    if validation_cache.is_dir():
        released += sum(p.stat().st_size for p in validation_cache.rglob("*") if p.is_file())
        shutil.rmtree(validation_cache)
    write_json(run / "storage_cleanup.json", {"selection_sha256": digest(run / "selection.json"),
        "metrics_sha256": digest(metrics_path), "selected_adapter_sha256": adapter_hashes(selected),
        "released_bytes": released, "full_checkpoints_remaining": 0})
    print(f"Cleaned training states; released {released / (1024 ** 3):.2f} GiB", flush=True)


def summarize(args: argparse.Namespace) -> None:
    prepared, root = args.prepared.resolve(), args.run_root.resolve()
    manifest = m3.check_prepared(prepared)
    rows = []
    metric = TEST_METRIC
    for target in runtime.CORPORA:
        run = root / f"target_{target}"
        result = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
        cleanup_marker = json.loads((run / "storage_cleanup.json").read_text(encoding="utf-8"))
        expected_n = manifest["fold_report"][target]["test_documents"]
        if (result.get("version") != m3.VERSION or result.get("target") != target or
                result.get("binding") != m3.binding(prepared) or result.get("n_test") != expected_n or
                result.get("selection") != selection or
                result.get("prediction_sha256") != digest(run / "predictions.jsonl") or
                cleanup_marker.get("full_checkpoints_remaining") != 0 or
                cleanup_marker.get("selection_sha256") != digest(run / "selection.json") or
                cleanup_marker.get("metrics_sha256") != digest(run / "metrics.json")):
            raise ValueError(f"Incomplete or mismatched selected result: {target}")
        rows.append({"target": target, "n_test": expected_n, "checkpoint": selection["candidate"],
                     "step": selection["step"], "threshold": selection["threshold"],
                     **result["metrics"], **result["counts"]})
    macros = {key: sum(row[key] for row in rows) / len(rows)
              for key in ("precision", "recall", metric)}
    rows.append({"target": "MACRO_6_CORPUS", "n_test": sum(r["n_test"] for r in rows),
                 "checkpoint": "source-selected per target", "step": "per target",
                 "threshold": "source-selected per target", **macros,
                 **{key: sum(r.get(key, 0) for r in rows) for key in m3.FIELDS}})
    runtime.write_csv(root / "summary.csv", rows,
        ("target", "n_test", "checkpoint", "step", "threshold", "precision", "recall", metric,
         *m3.FIELDS))
    write_json(root / "summary.json", {"version": m3.VERSION, "method": m3.METHOD,
        "binding": m3.binding(prepared), "seed": runtime.SEED,
        "normalization": "trim Unicode whitespace and punctuation at span edges only",
        "selection_protocol": campaign_binding(prepared), "results": rows,
        "completed_at_utc": runtime.stamp()})
    print(json.dumps({"summary": str(root / "summary.csv"), "macro": macros}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("smoke", smoke), ("fit", fit), ("summarize", summarize)):
        cmd = commands.add_parser(name)
        cmd.add_argument("--prepared", type=Path, default=m3.PREPARED)
        cmd.add_argument("--run-root", type=Path, required=True)
        if name in {"smoke", "fit"}:
            cmd.add_argument("--wtpsplit-repo", type=Path, required=True)
        if name == "fit":
            cmd.add_argument("--task-index", type=int, required=True)
        cmd.set_defaults(func=function)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
