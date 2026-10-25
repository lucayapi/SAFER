#!/usr/bin/env python3
"""One-seed SaT-LoRA checkpoint ablation with Nicollin held out of all sources.

For a non-Nicollin target, train and validate on the other four non-Nicollin
corpora. For the Nicollin target, train and validate on all five others.
The original target test narratives, model, optimizer settings, checkpoint
interval and exact-span scorer remain those of the elementary checkpoint run.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
from contextlib import closing
from pathlib import Path

import torch

import rq1_final_lodo as historical
import sat_elementary_checkpoint_lodo as prior
import sat_elementary_lodo as elementary
import sat_lodo_checkpoint_select as checkpoint
import sat_lodo_gold_v3 as base

VERSION = "sat_elementary_no_nicollin_v1"
METHOD = "SaT+LoRA elementary units; Nicollin excluded from sources"
EXCLUDED_SOURCE = "nicollin"
CURVE_FIELDS = prior.CURVE_FIELDS
METRICS = prior.METRICS


def code_digest() -> str:
    """Keep the run binding stable across Windows and Linux line endings."""
    source = Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def source_domains(target: str) -> tuple[str, ...]:
    if target not in base.CORPORA:
        raise ValueError(f"Unknown held-out target: {target}")
    return tuple(c for c in base.CORPORA if c not in {target, EXCLUDED_SOURCE})


def signature(bundle: Path) -> dict:
    return {"version": VERSION,
            "bundle_manifest_sha256": base.digest(bundle / "manifest.json"),
            "elementary_code_sha256": elementary.code_digest(),
            "checkpoint_code_sha256": base.digest(Path(prior.__file__)),
            "historical_code_sha256": base.digest(Path(historical.__file__)),
            "export_code_sha256": base.digest(Path(checkpoint.__file__)),
            "template_sha256": base.digest(base.legacy.TEMPLATE),
            "script_sha256": code_digest(),
            "excluded_source": EXCLUDED_SOURCE, "seed": base.SEED}


def prepared_signature(bundle: Path) -> dict:
    return {"version": VERSION, "bundle_manifest_sha256": base.digest(bundle / "manifest.json"),
            "elementary_code_sha256": elementary.code_digest(),
            "script_sha256": code_digest(),
            "excluded_source": EXCLUDED_SOURCE, "targets": list(base.CORPORA),
            "seed": base.SEED}


def check_prepared(bundle: Path, prepared: Path) -> dict:
    base.check_bundle(bundle)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed = json.loads((prepared / "complete.json").read_text(encoding="utf-8"))
    if (completed.get("manifest_sha256") != base.digest(manifest_path) or
            any(manifest.get(k) != v for k, v in prepared_signature(bundle).items())):
        raise ValueError("Filtered elementary dataset has changed")
    wanted = {f"folds/target_{target}/training.pth" for target in base.CORPORA}
    if set(manifest["files"]) != wanted:
        raise ValueError("Filtered dataset must contain exactly six folds")
    for target in base.CORPORA:
        report = manifest["source_report"][target]
        sources = source_domains(target)
        if (report["sources"] != list(sources) or
                set(report["train"]) != set(sources) or
                set(report["val"]) != set(sources)):
            raise ValueError(f"Wrong source domains for {target}")
    for relative, expected in manifest["files"].items():
        path = (prepared / relative).resolve()
        if (not path.is_relative_to(prepared.resolve()) or not path.is_file() or
                base.digest(path) != expected):
            raise ValueError(f"Changed filtered training data: {relative}")
    return manifest


def prepare(bundle: Path, prepared: Path) -> dict:
    base.check_bundle(bundle)
    if prepared.exists():
        return check_prepared(bundle, prepared)
    staging = prepared.with_name(prepared.name + ".preparing")
    if staging.exists():
        if staging.resolve().parent != prepared.resolve().parent:
            raise ValueError(f"Unsafe staging directory: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    files, report = {}, {}
    for target in base.CORPORA:
        sources = source_domains(target)
        fold = bundle / "folds" / f"target_{target}"
        examples, counts = {}, {}
        for partition in ("train", "val"):
            rows = base.read_jsonl(fold / f"{partition}.jsonl")
            selected = [row for row in rows if row["corpus"] in sources]
            if (len({(r["corpus"], r["document_id"]) for r in selected}) != len(selected) or
                    {row["corpus"] for row in selected} != set(sources) or
                    any(row["corpus"] == target for row in selected)):
                raise ValueError(f"Invalid {partition} split for {target}")
            chunks_list, source_counts = [], collections.defaultdict(collections.Counter)
            for row in selected:
                source_counts[row["corpus"]]["source_documents"] += 1
                chunks, reason, _ = elementary.elementary_chunks(row)
                if chunks is None:
                    source_counts[row["corpus"]][f"skipped_{reason}"] += 1
                    continue
                chunks_list.append(chunks)
                source_counts[row["corpus"]]["trainer_documents"] += 1
                source_counts[row["corpus"]]["facts"] += len(row["units"])
                source_counts[row["corpus"]]["excluded"] += len(row["excluded_spans"])
            if not chunks_list:
                raise ValueError(f"No trainable examples for {target}/{partition}")
            examples[partition] = chunks_list
            counts[partition] = {c: dict(source_counts[c]) for c in sources}
        data = {"fr": {"sentence": {base.DATASET_NAME: {
            "meta": {"train_data": examples["train"]}, "data": examples["val"]}}}}
        output = staging / "folds" / f"target_{target}" / "training.pth"
        output.parent.mkdir(parents=True)
        torch.save(data, output)
        files[output.relative_to(staging).as_posix()] = base.digest(output)
        report[target] = {"sources": list(sources), **counts}
        print(json.dumps({"target": target, "sources": sources,
                          "train_documents": len(examples["train"]),
                          "val_documents": len(examples["val"])}), flush=True)
    manifest = {**prepared_signature(bundle), "files": files,
                "source_report": report, "created_at_utc": base.stamp()}
    checkpoint.write_json_atomic(staging / "manifest.json", manifest)
    checkpoint.write_json_atomic(staging / "complete.json",
                                 {"manifest_sha256": base.digest(staging / "manifest.json")})
    check_prepared(bundle, staging)
    staging.rename(prepared)
    return manifest


def inputs(args):
    bundle, root = args.bundle.resolve(), args.run_root.resolve()
    prepared = root / "prepared"
    check_prepared(bundle, prepared)
    return bundle, prepared, root, signature(bundle)


def smoke(args):
    bundle, root = args.bundle.resolve(), args.run_root.resolve()
    prepare(bundle, root / "prepared")
    _, prepared, _, expected = inputs(args)
    marker = root / "smoke_ok.json"
    if marker.is_file():
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ValueError("Smoke marker belongs to a different experiment")
        historical.adapter_hashes(base.legacy.adapter_folder(root / "smoke" / "trainer"))
        return
    repo = args.wtpsplit_repo.resolve()
    base.legacy.git_revision(repo)
    base.legacy.gpu_runtime(repo)
    model_dir, tokenizer_dir = base.pinned_backbone(bundle)
    target = next(c for c in base.CORPORA if c != EXCLUDED_SOURCE)
    sample = torch.load(prepared / "folds" / f"target_{target}" / "training.pth",
                        map_location="cpu", weights_only=False)["fr"]["sentence"][base.DATASET_NAME]
    work = root / "smoke"
    elementary.bind_run(work, expected)
    small = {"fr": {"sentence": {base.DATASET_NAME: {
        "meta": {"train_data": sample["meta"]["train_data"][:8]},
        "data": sample["data"][:4]}}}}
    small_path = work / "training.pth"
    torch.save(small, small_path)
    trainer = work / "trainer"
    config = base.trainer_config(bundle, target, trainer, model_dir, tokenizer_dir)
    config.update({"text_path": str(small_path), "num_train_epochs": 1,
                   "max_steps": 1, "save_steps": 1, "save_total_limit": None,
                   "per_device_train_batch_size": 2, "per_device_eval_batch_size": 2,
                   "gradient_accumulation_steps": 1})
    if not (trainer / "checkpoint-1" / "pytorch_model.bin").is_file():
        base.legacy.run_trainer(repo, config, work / "config.json")
    final = base.legacy.adapter_folder(trainer)
    historical.adapter_hashes(final)
    first = trainer / "checkpoint-1"
    exported = work / "checkpoint_adapter"
    checkpoint.export_checkpoint_adapter(first, exported, final)
    historical.adapter_hashes(exported)
    state_path = trainer / "trainer_state.json"
    if int(json.loads(state_path.read_text(encoding="utf-8"))["global_step"]) < 2:
        config.update({"max_steps": 2, "resume_from_checkpoint": str(first)})
        checkpoint.run_upstream(repo, config, work / "resume_config.json")
    if int(json.loads(state_path.read_text(encoding="utf-8"))["global_step"]) != 2:
        raise ValueError("Smoke optimizer resume failed")
    second = work / "resumed_checkpoint_adapter"
    checkpoint.export_checkpoint_adapter(trainer / "checkpoint-2", second, final)
    saved = torch.load(final / "pytorch_adapter.bin", map_location="cpu", weights_only=True)
    resumed = torch.load(second / "pytorch_adapter.bin", map_location="cpu", weights_only=True)
    if set(saved) != set(resumed) or any(not torch.equal(saved[k], resumed[k]) for k in saved):
        raise ValueError("Resumed adapter differs from final adapter")
    row = next(row for row in base.read_jsonl(
        bundle / "folds" / f"target_{target}" / "val.jsonl")
        if row["corpus"] in source_domains(target))
    model = base.load_pinned_sat(model_dir, tokenizer_dir, final)
    base.legacy.pieces_to_spans(
        row["raw_text"], model.split(row["raw_text"], threshold=base.DEFAULT))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    checkpoint.write_json_atomic(marker, expected)
    print(f"Filtered-source GPU smoke passed: {marker}", flush=True)


def validation_curve(rows: list[dict], db, candidate: dict, sources: tuple[str, ...]):
    if {row["corpus"] for row in rows} != set(sources) or EXCLUDED_SOURCE in sources:
        raise ValueError("Validation includes forbidden source or misses a source domain")
    counts = collections.defaultdict(collections.Counter)
    for row in rows:
        probs = base.get_probs(db, row)
        for threshold in base.THRESHOLDS:
            predicted = base.legacy.spans_from_probabilities(row["raw_text"], probs, threshold)
            counts[row["corpus"], threshold].update(elementary.score(row, predicted))
    curve = []
    for threshold in base.THRESHOLDS:
        values = [elementary.rates(counts[source, threshold]) for source in sources]
        if any(counts[source, threshold]["excluded_ref"] == 0 for source in sources):
            raise ValueError("Source validation domain lacks excluded spans")
        curve.append({"candidate": candidate["candidate"], "step": candidate["step"],
                      "epoch": candidate["epoch"], "threshold": threshold,
                      "macro_elementary_exact_f1": sum(v["elementary_exact_f1"] for v in values) / len(values),
                      "macro_factual_exact_recall": sum(v["factual_exact_recall"] for v in values) / len(values),
                      "macro_excluded_exact_recall": sum(v["excluded_exact_recall"] for v in values) / len(values),
                      "selected": False})
    return curve


def evaluate_target(bundle, target, run, selection, model_dir, tokenizer_dir, expected):
    rows = base.read_jsonl(bundle / "folds" / f"target_{target}" / "test.jsonl")
    adapter = run / "selected_adapter"
    cache = run / "test_probabilities.sqlite3"
    prior.infer_to_cache(adapter, rows, cache,
                         {"experiment": expected, "weights": selection["weights"],
                          "candidate": selection["candidate"], "partition": "target_test"},
                         model_dir, tokenizer_dir)
    model = base.load_pinned_sat(model_dir, tokenizer_dir, adapter)
    with closing(base._probability_connection(cache)) as db:
        base.verify_native(model, db, rows, selection["threshold"])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    counts = collections.Counter()
    output = run / "predictions.jsonl"
    temporary = output.with_name(output.name + ".tmp")
    with closing(base._probability_connection(cache)) as db, temporary.open(
            "w", encoding="utf-8") as stream:
        for row in rows:
            pred = base.legacy.spans_from_probabilities(
                row["raw_text"], base.get_probs(db, row), selection["threshold"])
            counts.update(elementary.score(row, pred))
            stream.write(json.dumps({"corpus": target, "document_id": row["document_id"],
                                     "threshold": selection["threshold"],
                                     "segments": [{"start": a, "end": b} for a, b in pred]},
                                    ensure_ascii=False) + "\n")
    temporary.replace(output)
    metrics = {"version": VERSION, "method": METHOD, "target": target,
               "n_test": len(rows), "seed": base.SEED,
               "candidate": selection["candidate"], "step": selection["step"],
               "threshold": selection["threshold"], "sources": list(source_domains(target)),
               "selection_sha256": base.digest(run / "selection.json"),
               "predictions_sha256": base.digest(output),
               "counts": dict(counts), "metrics": elementary.rates(counts),
               "completed_at_utc": base.stamp()}
    checkpoint.write_json_atomic(run / "metrics.json", metrics)
    print(json.dumps({"target": target, **metrics["metrics"]}), flush=True)


def cleanup(run: Path, selection: dict):
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    if (metrics["version"] != VERSION or metrics["target"] != run.name.removeprefix("target_") or
            metrics["selection_sha256"] != base.digest(run / "selection.json") or
            metrics["predictions_sha256"] != base.digest(run / "predictions.jsonl") or
            historical.adapter_hashes(run / "selected_adapter") != selection["weights"]):
        raise ValueError("Cannot clean an unverified run")
    trainer = run / "trainer"
    released = 0
    for folder in trainer.glob("checkpoint-*"):
        if folder.is_dir() and folder.name.removeprefix("checkpoint-").isdigit():
            released += historical.remove_checkpoint(trainer, folder)
    checkpoint.write_json_atomic(run / "storage_cleanup.json",
                                 {"released_bytes": released, "full_checkpoints_remaining": 0,
                                  "selected_weights": selection["weights"]})


def fit(args):
    bundle, prepared, root, expected = inputs(args)
    if json.loads((root / "smoke_ok.json").read_text(encoding="utf-8")) != expected:
        raise RuntimeError("Run the no-Nicollin GPU smoke first")
    if not 0 <= args.task_index < len(base.CORPORA):
        raise ValueError("task-index must be 0..5")
    target = base.CORPORA[args.task_index]
    sources = source_domains(target)
    repo = args.wtpsplit_repo.resolve()
    base.legacy.git_revision(repo)
    base.legacy.gpu_runtime(repo)
    model_dir, tokenizer_dir = base.pinned_backbone(bundle)
    run = root / f"target_{target}"
    elementary.bind_run(run, expected)
    if (run / "metrics.json").is_file():
        selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
        cleanup(run, selection)
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
        historical.run_trainer_with_harvest(repo, config, run / "config.json", run,
                                            base.legacy.adapter_folder(root / "smoke" / "trainer"))
    historical.harvest_checkpoints(run, trainer,
                                   base.legacy.adapter_folder(root / "smoke" / "trainer"))
    candidates = historical.candidates_from_exports(run, trainer)
    base.legacy.export_loss_history(trainer / "trainer_state.json", run / "loss_history.csv")
    checkpoint.write_json_atomic(run / "training_done.json",
                                 {"experiment": expected, "target": target,
                                  "sources": list(sources), "seed": base.SEED,
                                  "checkpoint_steps": [c["step"] for c in candidates],
                                  "completed_at_utc": base.stamp()})
    source_rows = [row for row in base.read_jsonl(
        bundle / "folds" / f"target_{target}" / "val.jsonl") if row["corpus"] in sources]
    if {row["corpus"] for row in source_rows} != set(sources):
        raise ValueError("Incomplete source validation")
    curves, adapters = [], {}
    for candidate in candidates:
        name, adapter = candidate["candidate"], candidate["adapter"]
        weights = historical.adapter_hashes(adapter)
        cache = run / "validation_cache" / f"{name}.sqlite3"
        prior.infer_to_cache(adapter, source_rows, cache,
                             {"experiment": expected, "weights": weights,
                              "candidate": name, "partition": "source_validation"},
                             model_dir, tokenizer_dir)
        with closing(base._probability_connection(cache)) as db:
            curve = validation_curve(source_rows, db, candidate, sources)
        curves.extend(curve)
        adapters[name] = adapter
        base.write_csv(run / f"source_validation_{name}.csv", curve, CURVE_FIELDS)
    chosen = prior.best_candidate(curves)
    chosen["selected"] = True
    base.write_csv(run / "checkpoint_validation.csv", curves, CURVE_FIELDS)
    selection = {"experiment": expected, "target": target, "sources": list(sources),
                 "criterion": f"macro exact elementary F1 over {len(sources)} non-Nicollin source domains",
                 "candidate": chosen["candidate"], "step": chosen["step"],
                 "epoch": chosen["epoch"], "threshold": chosen["threshold"],
                 "source_macro_elementary_exact_f1": chosen["macro_elementary_exact_f1"],
                 "weights": historical.adapter_hashes(adapters[chosen["candidate"]]),
                 "curve_sha256": base.digest(run / "checkpoint_validation.csv")}
    path = run / "selection.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != selection:
            raise ValueError("Source selection changed on resume")
    else:
        checkpoint.write_json_atomic(path, selection)
    historical.persist_selected_adapter(adapters[chosen["candidate"]],
                                        run / "selected_adapter", selection["weights"])
    evaluate_target(bundle, target, run, selection, model_dir, tokenizer_dir, expected)
    cleanup(run, selection)


def summarize(args):
    bundle, prepared, root, expected = inputs(args)
    manifest = base.check_bundle(bundle)
    rows = []
    for target in base.CORPORA:
        run = root / f"target_{target}"
        selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        if (json.loads((run / "binding.json").read_text(encoding="utf-8")) != expected or
                selection["experiment"] != expected or metrics["version"] != VERSION or
                metrics["target"] != target or metrics["sources"] != list(source_domains(target)) or
                metrics["n_test"] != manifest["source_report"][target]["gold_documents"] or
                metrics["selection_sha256"] != base.digest(run / "selection.json") or
                metrics["predictions_sha256"] != base.digest(run / "predictions.jsonl") or
                selection["curve_sha256"] != base.digest(run / "checkpoint_validation.csv") or
                historical.adapter_hashes(run / "selected_adapter") != selection["weights"]):
            raise ValueError(f"Incomplete ablation fold: {target}")
        rows.append({"target": target, "n_test": metrics["n_test"],
                     "n_sources": len(source_domains(target)),
                     "candidate": metrics["candidate"], "step": metrics["step"],
                     "threshold": metrics["threshold"], **metrics["metrics"], **metrics["counts"]})
    fold_rows = list(rows)
    aggregates = {}
    for label, subset in (("MACRO_6_CORPUS", fold_rows),
                          ("MACRO_5_NON_NICOLLIN", [r for r in fold_rows
                                                    if r["target"] != EXCLUDED_SOURCE])):
        macro = {key: sum(row[key] for row in subset) / len(subset) for key in METRICS}
        aggregates[label] = macro
        rows.append({"target": label, "n_test": sum(row["n_test"] for row in subset),
                     "n_sources": "varies", "candidate": "source_selected",
                     "step": "source_selected", "threshold": "source_selected", **macro,
                     **{key: sum(row.get(key, 0) for row in subset)
                        for key in elementary.COUNT_FIELDS}})
    base.write_csv(root / "summary.csv", rows,
                   ("target", "n_test", "n_sources", "candidate", "step", "threshold",
                    *METRICS, *elementary.COUNT_FIELDS))
    checkpoint.write_json_atomic(root / "summary.json",
                                 {"version": VERSION, "method": METHOD,
                                  "experiment": expected, "results": rows,
                                  "aggregates": aggregates, "completed_at_utc": base.stamp()})
    print(json.dumps({"summary": str(root / "summary.csv"), "aggregates": aggregates}),
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, func in (("prepare", None), ("check", None), ("smoke", smoke),
                       ("fit", fit), ("summarize", summarize)):
        cmd = commands.add_parser(name)
        cmd.add_argument("--bundle", type=Path, default=elementary.BASE_BUNDLE)
        cmd.add_argument("--run-root", type=Path, required=True)
        if name in {"smoke", "fit"}:
            cmd.add_argument("--wtpsplit-repo", type=Path, required=True)
        if name == "fit":
            cmd.add_argument("--task-index", type=int, required=True)
        cmd.set_defaults(func=func)
    args = parser.parse_args()
    if args.command == "prepare":
        report = prepare(args.bundle.resolve(), args.run_root.resolve() / "prepared")
        print(json.dumps({"prepared": True, "targets": report["targets"]}), flush=True)
    elif args.command == "check":
        _, _, _, expected = inputs(args)
        print(json.dumps({"ready": True, "experiment": expected}), flush=True)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
