#!/usr/bin/env python3
"""RQ1 SaT-LoRA rerun: select checkpoint and threshold on source validation.

This is a separate experiment from sat_lodo_gold_v3.py.  The target corpus is
not read for model selection.  Only after selection do we evaluate it once.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import sat_lodo_gold_v3 as base


SCRIPT = Path(__file__).resolve()
RESUME_SCRIPT = SCRIPT.with_name("run_sat_lora_resume.py")
RUN_NAME = "sat12l_sm_gold_v3_rq1_checkpoint_selection_v1"
SAVE_STEPS = 500
CURVE_FIELDS = ("candidate", "step", "epoch", "threshold", "source_macro_span_f1",
                *(f"source_f1_{corpus}" for corpus in base.CORPORA))


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def signature(bundle: Path) -> dict[str, str]:
    return {"bundle_manifest_sha256": base.digest(bundle / "manifest.json"),
            "template_sha256": base.digest(base.legacy.TEMPLATE),
            "base_experiment_sha256": base.digest(Path(base.__file__)),
            "segmentation_helper_sha256": base.digest(Path(base.legacy.__file__)),
            "script_sha256": base.digest(SCRIPT),
            "resume_script_sha256": base.digest(RESUME_SCRIPT)}


def bind_run(run: Path, bundle: Path) -> None:
    expected = signature(bundle)
    path = run / "checkpoint_selection_binding.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Run belongs to another bundle or code version: {run}")
    elif run.exists() and any(run.iterdir()):
        raise ValueError(f"Unbound run directory: {run}")
    else:
        run.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, expected)


def run_upstream(repo: Path, config: dict, path: Path) -> None:
    if "resume_from_checkpoint" not in config:
        base.legacy.run_trainer(repo, config, path)
        return
    base.write_json(path, config)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["WANDB_DISABLED"] = "true"
    env["TOKENIZERS_PARALLELISM"] = "false"
    subprocess.run([sys.executable, "-u", str(RESUME_SCRIPT), str(path)],
                   cwd=repo, env=env, check=True)


def _save_torch_atomic(path: Path, value: dict) -> None:
    import torch

    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def export_checkpoint_adapter(checkpoint: Path, adapter_dir: Path,
                              final_adapter: Path) -> str:
    """Convert the official full-state checkpoint into its saved LoRA format."""
    import torch

    source = checkpoint / "pytorch_model.bin"
    if not source.is_file():
        raise FileNotFoundError(source)
    fingerprint = base.digest(source)
    marker = adapter_dir / "checkpoint_source.json"
    if marker.is_file():
        prior = json.loads(marker.read_text(encoding="utf-8"))
        if prior.get("checkpoint_sha256") != fingerprint:
            raise ValueError(f"Checkpoint changed after adapter export: {source}")
        for name, expected in prior["files"].items():
            if base.digest(adapter_dir / name) != expected:
                raise ValueError(f"Incomplete exported adapter: {adapter_dir / name}")
        return fingerprint

    adapter_dir.mkdir(parents=True, exist_ok=True)
    state = torch.load(source, map_location="cpu", weights_only=True)
    adapter = {key.removeprefix("backbone."): value for key, value in state.items()
               if key.startswith("backbone.") and ".loras.text." in key}
    head = {name: state["backbone." + name]
            for name in ("classifier.weight", "classifier.bias")}
    expected_adapter = torch.load(final_adapter / "pytorch_adapter.bin",
                                  map_location="cpu", weights_only=True)
    expected_head = torch.load(final_adapter / "pytorch_model_head.bin",
                               map_location="cpu", weights_only=True)
    if (set(adapter) != set(expected_adapter) or set(head) != set(expected_head) or
            any(adapter[key].shape != value.shape for key, value in expected_adapter.items()) or
            any(head[key].shape != value.shape for key, value in expected_head.items())):
        raise ValueError(f"Checkpoint is incompatible with the final adapter: {checkpoint}")
    for name in ("adapter_config.json", "head_config.json"):
        shutil.copy2(final_adapter / name, adapter_dir / name)
    _save_torch_atomic(adapter_dir / "pytorch_adapter.bin", adapter)
    _save_torch_atomic(adapter_dir / "pytorch_model_head.bin", head)
    files = {name: base.digest(adapter_dir / name) for name in
             ("adapter_config.json", "head_config.json", "pytorch_adapter.bin",
              "pytorch_model_head.bin")}
    write_json_atomic(marker, {"checkpoint": str(checkpoint),
                               "checkpoint_sha256": fingerprint, "files": files})
    return fingerprint


def complete_adapter(path: Path) -> bool:
    return all((path / name).is_file() for name in
               ("adapter_config.json", "head_config.json", "pytorch_adapter.bin",
                "pytorch_model_head.bin"))


def latest_resumable_checkpoint(trainer: Path) -> Path | None:
    """Ignore an unfinished checkpoint left by an interrupted Slurm task."""
    folders = sorted((folder for folder in trainer.glob("checkpoint-*")
                      if folder.is_dir() and folder.name.removeprefix("checkpoint-").isdigit()),
                     key=lambda folder: int(folder.name.removeprefix("checkpoint-")), reverse=True)
    required = ("pytorch_model.bin", "optimizer.pt", "scheduler.pt", "trainer_state.json")
    return next((folder for folder in folders
                 if all((folder / name).is_file() and (folder / name).stat().st_size > 0
                        for name in required)), None)


def training_candidates(trainer: Path) -> list[dict]:
    state_path = trainer / "trainer_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Training did not save its final state: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    final_step = int(state["global_step"])
    if final_step < 1:
        raise ValueError("Training stopped without an optimizer step")
    checkpoints = {}
    for folder in trainer.glob("checkpoint-*"):
        if folder.is_dir() and folder.name.removeprefix("checkpoint-").isdigit():
            checkpoints[int(folder.name.removeprefix("checkpoint-"))] = folder
    expected = set(range(SAVE_STEPS, final_step + 1, SAVE_STEPS))
    if not expected.issubset(checkpoints):
        raise ValueError(f"Missing checkpoints; save_total_limit must be null: "
                         f"{sorted(expected - checkpoints.keys())}")
    result = []
    for step in sorted(expected):
        folder = checkpoints[step]
        if not (folder / "pytorch_model.bin").is_file():
            raise FileNotFoundError(folder / "pytorch_model.bin")
        if step == final_step:
            continue  # The final adapter has exactly these weights.
        snapshot = json.loads((folder / "trainer_state.json").read_text(encoding="utf-8"))
        result.append({"candidate": f"checkpoint-{step}", "step": step,
                       "epoch": float(snapshot["epoch"]), "checkpoint": folder})
    result.append({"candidate": "final", "step": final_step,
                   "epoch": float(state["epoch"]), "checkpoint": None})
    return result


def validation_curve(rows: list[dict], db: sqlite3.Connection,
                     candidate: str, step: int, epoch: float) -> list[dict]:
    sources = {row["corpus"] for row in rows}
    if len(sources) != 5:
        raise ValueError("Checkpoint validation must contain five source corpora")
    totals = collections.defaultdict(collections.Counter)
    for row in rows:
        probs = base.get_probs(db, row)
        for threshold in base.THRESHOLDS:
            spans = base.legacy.spans_from_probabilities(row["raw_text"], probs, threshold)
            totals[row["corpus"], threshold].update(base.score(row, spans))
    curve = []
    for threshold in base.THRESHOLDS:
        per_source = {source: base.prf(totals[source, threshold]["span_tp"],
                                       totals[source, threshold]["span_pred"],
                                       totals[source, threshold]["span_ref"])[2]
                      for source in sources}
        if any(totals[source, threshold]["span_ref"] == 0 for source in sources):
            raise ValueError("Validation source has no factual reference units")
        curve.append({"candidate": candidate, "step": step, "epoch": epoch,
                      "threshold": threshold,
                      "source_macro_span_f1": sum(per_source.values()) / len(sources),
                      **{f"source_f1_{source}": per_source.get(source) for source in base.CORPORA}})
    return curve


def best_row(curve: list[dict]) -> dict:
    """Prefer the highest source F1; deterministic ties favor default then early."""
    if not curve:
        raise ValueError("No checkpoint/threshold candidates")
    return min(curve, key=lambda row: (-row["source_macro_span_f1"],
                                       abs(row["threshold"] - base.DEFAULT),
                                       row["step"], row["threshold"]))


def smoke(args: argparse.Namespace) -> None:
    import torch

    bundle = Path(args.bundle).resolve()
    base.check_bundle(bundle)
    repo = Path(args.wtpsplit_repo).resolve()
    base.legacy.git_revision(repo)
    base.legacy.gpu_runtime(repo)
    run_root = Path(args.run_root).resolve()
    expected = signature(bundle)
    marker = run_root / "checkpoint_smoke_ok.json"
    if marker.is_file():
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ValueError("Checkpoint smoke belongs to another bundle or code version")
        print(f"Checkpoint smoke already passed: {marker}", flush=True)
        return
    base.smoke(args)
    folder = run_root / "smoke" / "trainer"
    final_adapter = base.legacy.adapter_folder(folder)
    checkpoint = folder / "checkpoint-1"
    export_checkpoint_adapter(checkpoint, run_root / "smoke" / "checkpoint_adapter",
                              final_adapter)
    config = json.loads((run_root / "smoke" / "config.json").read_text(encoding="utf-8"))
    config.update({"max_steps": 2, "resume_from_checkpoint": str(checkpoint),
                   "save_total_limit": None})
    state_path = folder / "trainer_state.json"
    if json.loads(state_path.read_text(encoding="utf-8"))["global_step"] < 2:
        run_upstream(repo, config, run_root / "smoke" / "resume_config.json")
    if json.loads(state_path.read_text(encoding="utf-8"))["global_step"] != 2:
        raise AssertionError("Official optimizer/scheduler resume did not reach step 2")
    newer = folder / "checkpoint-2"
    extracted = run_root / "smoke" / "resumed_checkpoint_adapter"
    export_checkpoint_adapter(newer, extracted, final_adapter)
    saved = torch.load(final_adapter / "pytorch_adapter.bin", map_location="cpu", weights_only=True)
    resumed = torch.load(extracted / "pytorch_adapter.bin", map_location="cpu", weights_only=True)
    if set(saved) != set(resumed) or not all(torch.equal(saved[key], resumed[key]) for key in saved):
        raise AssertionError("Resumed checkpoint differs from final saved adapter")
    write_json_atomic(marker, expected)
    print(f"Checkpoint export and optimizer-state resume passed: {marker}", flush=True)


def fit(args: argparse.Namespace) -> None:
    import torch

    bundle = Path(args.bundle).resolve()
    manifest = base.check_bundle(bundle)
    if not 0 <= args.task_index < len(base.CORPORA):
        raise ValueError("task-index must be 0..5")
    target = manifest["fold_order"][args.task_index]
    root = Path(args.run_root).resolve()
    smoke_marker = root / "checkpoint_smoke_ok.json"
    if not smoke_marker.is_file() or json.loads(smoke_marker.read_text(encoding="utf-8")) != signature(bundle):
        raise RuntimeError("Run the checkpoint-selection smoke job first")
    repo = Path(args.wtpsplit_repo).resolve()
    revision = base.legacy.git_revision(repo)
    runtime = base.legacy.gpu_runtime(repo)
    model_dir, tokenizer_dir = base.pinned_backbone(bundle)
    run = root / f"target_{target}"
    bind_run(run, bundle)
    trainer = run / "trainer"
    config = base.trainer_config(bundle, target, trainer, model_dir, tokenizer_dir)
    config["save_steps"] = SAVE_STEPS
    config["save_total_limit"] = None
    config["load_best_model_at_end"] = False
    final_adapter = base.legacy.adapter_folder(trainer)
    if not complete_adapter(final_adapter):
        checkpoint = latest_resumable_checkpoint(trainer)
        if checkpoint:
            config["resume_from_checkpoint"] = str(checkpoint)
        run_upstream(repo, config, run / "config.json")
    if not complete_adapter(final_adapter):
        raise RuntimeError(f"Final adapter not saved: {final_adapter}")
    candidates = training_candidates(trainer)
    state = trainer / "trainer_state.json"
    if state.is_file():
        base.legacy.export_loss_history(state, run / "loss_history.csv")
    write_json_atomic(run / "training_done.json", {
        "target": target, "reference": base.REFERENCE, "seed": base.SEED,
        "bundle_manifest_sha256": base.digest(bundle / "manifest.json"),
        "training_config_sha256": base.digest(run / "config.json"),
        "model_revision": model_dir.name, "tokenizer_revision": tokenizer_dir.name,
        "wtpsplit_revision": revision, "runtime": runtime,
        "checkpoint_steps": [candidate["step"] for candidate in candidates],
        "final_adapter": str(final_adapter)})

    # Target labels are checked for bundle integrity, but never used for selection.
    folder = bundle / "folds" / f"target_{target}"
    val = base.read_jsonl(folder / "val.jsonl")
    all_rows = []
    adapters = {}
    cache_paths = {}
    for candidate in candidates:
        name = candidate["candidate"]
        if candidate["checkpoint"] is None:
            adapter = final_adapter
            fingerprint = hashlib.sha256(
                (base.digest(adapter / "pytorch_adapter.bin") +
                 base.digest(adapter / "pytorch_model_head.bin")).encode()).hexdigest()
        else:
            adapter = run / "checkpoint_adapters" / name
            fingerprint = export_checkpoint_adapter(candidate["checkpoint"], adapter, final_adapter)
        cache = run / "checkpoint_validation_cache" / f"{name}.sqlite3"
        cache_marker = cache.with_suffix(".json")
        expected = {"candidate": name, "fingerprint": fingerprint,
                    "bundle_manifest_sha256": base.digest(bundle / "manifest.json")}
        if cache_marker.is_file():
            if json.loads(cache_marker.read_text(encoding="utf-8")) != expected:
                raise ValueError(f"Validation cache belongs to different weights: {cache}")
        elif cache.exists():
            raise ValueError(f"Unbound validation cache: {cache}")
        else:
            write_json_atomic(cache_marker, expected)
        print(f"Validating {target}/{name} at step {candidate['step']}", flush=True)
        with closing(base._probability_connection(cache, create=True)) as db:
            present = db.execute("SELECT COUNT(*) FROM probabilities").fetchone()[0]
            if present < len(val):
                sat = base.load_pinned_sat(model_dir, tokenizer_dir, adapter)
                base.infer_adapted(sat, db, val)
                del sat
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if db.execute("SELECT COUNT(*) FROM probabilities").fetchone()[0] != len(val):
                raise ValueError("Validation cache has missing or extra documents")
            curve = validation_curve(val, db, name, candidate["step"], candidate["epoch"])
        all_rows.extend(curve)
        adapters[name] = adapter
        cache_paths[name] = cache
        base.write_csv(run / f"source_validation_{name}.csv", curve, CURVE_FIELDS)
    base.write_csv(run / "checkpoint_validation.csv", all_rows, CURVE_FIELDS)
    selected = best_row(all_rows)
    with closing(base._probability_connection(bundle / "base_probabilities.sqlite3")) as db:
        base_curve = validation_curve(val, db, "M1_base", -1, -1.0)
    m1 = best_row(base_curve)
    base.write_csv(run / "source_validation_M1.csv", base_curve, CURVE_FIELDS)
    selection = {"target": target, "reference": base.REFERENCE,
                 "bundle_manifest_sha256": base.digest(bundle / "manifest.json"),
                 "criterion": "macro of five source-corpus micro exact-span F1s",
                 "tie_break": "threshold nearest 0.25, earlier optimizer step, then smaller threshold",
                 "n_source_validation": len(val),
                 "M1_threshold": m1["threshold"], "M1_source_macro_span_f1": m1["source_macro_span_f1"],
                 "M2_candidate": selected["candidate"], "M2_step": selected["step"],
                 "M2_epoch": selected["epoch"], "M2_threshold": selected["threshold"],
                 "M2_source_macro_span_f1": selected["source_macro_span_f1"],
                 "adapter": str(adapters[selected["candidate"]]),
                 "checkpoint_validation_sha256": base.digest(run / "checkpoint_validation.csv")}
    selection_path = run / "checkpoint_selection.json"
    if selection_path.is_file():
        if json.loads(selection_path.read_text(encoding="utf-8")) != selection:
            raise ValueError("Previously chosen checkpoint differs from current source validation")
    else:
        write_json_atomic(selection_path, selection)
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)

    selected_cache = cache_paths[selected["candidate"]]
    output_cache = run / "adapted_probabilities.sqlite3"
    output_marker = run / "adapted_probabilities.selection.json"
    cache_binding = {"candidate": selected["candidate"],
                     "source_sha256": base.digest(selected_cache)}
    if output_marker.is_file():
        if json.loads(output_marker.read_text(encoding="utf-8")) != cache_binding:
            raise ValueError("Target probability cache belongs to another selected checkpoint")
        if not output_cache.is_file():
            shutil.copy2(selected_cache, output_cache)
    elif output_cache.exists():
        raise ValueError("Unbound target probability cache")
    else:
        shutil.copy2(selected_cache, output_cache)
        write_json_atomic(output_marker, cache_binding)
    metrics_path = run / "metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        base.evaluate_fold(bundle, target, run, adapters[selected["candidate"]],
                           model_dir, tokenizer_dir)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (metrics["selection"]["M1_threshold"] != m1["threshold"] or
            metrics["selection"]["M2_threshold"] != selected["threshold"]):
        raise AssertionError("Target evaluation did not reuse source-selected thresholds")
    metrics["selection"]["M2_checkpoint"] = selected["candidate"]
    metrics["selection"]["M2_checkpoint_step"] = selected["step"]
    metrics["selection"]["M2_source_macro_span_f1"] = selected["source_macro_span_f1"]
    write_json_atomic(metrics_path, metrics)
    print(f"Completed {target}: selected {selected['candidate']} at threshold "
          f"{selected['threshold']}; target F1={metrics['test']['M2']['span_f1']:.6f}", flush=True)


def summarize(args: argparse.Namespace) -> None:
    bundle = Path(args.bundle).resolve()
    base.check_bundle(bundle)
    root = Path(args.run_root).resolve()
    selections = []
    for target in base.CORPORA:
        path = root / f"target_{target}" / "checkpoint_selection.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        row = json.loads(path.read_text(encoding="utf-8"))
        selections.append({key: row[key] for key in
                           ("target", "M1_threshold", "M2_candidate", "M2_step",
                            "M2_epoch", "M2_threshold", "M2_source_macro_span_f1")})
    base.summarize(argparse.Namespace(bundle=str(bundle), run_root=str(root),
                                      bootstrap=args.bootstrap))
    base.write_csv(root / "checkpoint_selection_summary.csv", selections,
                   tuple(selections[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("smoke", smoke), ("fit", fit), ("summarize", summarize)):
        cmd = commands.add_parser(name)
        cmd.add_argument("--bundle", required=True)
        cmd.add_argument("--run-root", required=True)
        if name != "summarize":
            cmd.add_argument("--wtpsplit-repo", required=True)
        if name == "fit":
            cmd.add_argument("--task-index", type=int, required=True)
        if name == "summarize":
            cmd.add_argument("--bootstrap", type=int, default=1000)
        cmd.set_defaults(func=function)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
