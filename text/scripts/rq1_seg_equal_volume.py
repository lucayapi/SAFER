#!/usr/bin/env python3
"""Paired, source-only equal-character-budget SaT LoRA transfer experiment.

The existing single-source fits are immutable comparators.  For each of their
30 off-diagonal transfers this campaign trains one five-source model with the
same number of training characters (within 1%) and optimizer updates. Both
models select checkpoint and threshold on the *same single-source validation*
set, then score the same untouched target test narratives.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from contextlib import closing
from pathlib import Path

import rq1_final as final_data
import rq1_final_lodo as lodo
import sat_indomain_v3 as single
import sat_lodo_gold_v3 as sat

NAME = "rq1_seg_equal_volume_v2"
REFERENCE_NAME = "rq1_seg_equal_volume_reference_v1"
SEED = 271828
BOOTSTRAPS = 2000
def pair_name(target: str, source: str) -> str:
    return f"target_{target}__source_{source}"


def key(row: dict) -> tuple[str, str]:
    return row["corpus"], row["document_id"]


def input_chars(item: tuple[dict, list[str]]) -> int:
    """Exact length of the official trainer's newline-joined text copy."""
    return sum(map(len, item[1])) + max(0, len(item[1]) - 1)


def ordered(rows: list[tuple[dict, list[str]]], target: str, source: str,
            corpus: str) -> list[tuple[dict, list[str]]]:
    return sorted(rows, key=lambda item: hashlib.sha256(
        f"{SEED}\0{target}\0{source}\0{corpus}\0{item[0]['document_id']}".encode()
    ).hexdigest())


def balanced_sample(eligible: dict[str, list[tuple[dict, list[str]]]],
                    target: str, source: str, budget_chars: int) -> list[tuple[dict, list[str]]]:
    """Balance five sources by characters; match the mono character budget."""
    domains = [corpus for corpus in sat.CORPORA if corpus != target]
    pools = {corpus: ordered(eligible[corpus], target, source, corpus)
             for corpus in domains}
    if (min(map(len, pools.values())) == 0 or
            sum(input_chars(item) for pool in pools.values() for item in pool) < budget_chars):
        raise ValueError(f"Insufficient eligible source examples for {target}/{source}")
    selected, positions = [], {corpus: 0 for corpus in domains}
    used = {corpus: 0 for corpus in domains}
    total = 0
    while total < budget_chars:
        remaining = budget_chars - total
        options = [(used[corpus], corpus) for corpus in domains
                   if positions[corpus] < len(pools[corpus])]
        if not options:
            break
        _, corpus = min(options)
        item = pools[corpus][positions[corpus]]
        if input_chars(item) > remaining:
            # Final shortfall: find the closest still-unused narrative without
            # exceeding the character budget.  The previous balanced prefix
            # remains intact and the final adjustment is fully deterministic.
            fitting = [(input_chars(pools[c][index]), c, index)
                       for c in domains for index in range(positions[c], len(pools[c]))
                       if input_chars(pools[c][index]) <= remaining]
            if not fitting:
                break
            _, corpus, index = max(fitting)
            pools[corpus][positions[corpus]], pools[corpus][index] = (
                pools[corpus][index], pools[corpus][positions[corpus]])
            item = pools[corpus][positions[corpus]]
        selected.append(item)
        positions[corpus] += 1
        length = input_chars(item)
        used[corpus] += length
        total += length
    if (any(used[corpus] == 0 for corpus in domains)
            or abs(total - budget_chars) > max(1, int(budget_chars * .01))):
        raise ValueError(f"Could not match five-source character budget: {target}/{source} "
                         f"({total} vs {budget_chars})")
    return selected


def comparator_adapter(runs: Path, source: str) -> Path:
    run = runs / f"source_{source}"
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    if selection["M2_candidate"] == "final":
        return sat.legacy.adapter_folder(run / "trainer")
    return run / "checkpoint_adapters" / selection["M2_candidate"]


def source_run_info(source_bundle: Path, runs: Path, source: str) -> dict:
    run = runs / f"source_{source}"
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    config_path, state_path = run / "config.json", run / "trainer" / "trainer_state.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    done_path = run / "training_done.json"
    done = json.loads(done_path.read_text(encoding="utf-8"))
    runtime = json.loads((source_bundle / "rq1_runtime.json").read_text(encoding="utf-8"))
    steps = int(done["checkpoint_steps"][-1])
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if int(state["global_step"]) != steps:
            raise ValueError(f"Single-source final step differs from training log: {source}")
    adapter = comparator_adapter(runs, source)
    if (selection["source"] != source or metrics["source"] != source
            or selection["bundle_manifest_sha256"] != sat.digest(source_bundle / "manifest.json")
            or metrics["selection_sha256"] != sat.digest(run / "selection.json")
            or sat.digest(adapter / "pytorch_adapter.bin") != metrics["adapter_sha256"]
            or sat.digest(adapter / "pytorch_model_head.bin") != metrics["head_sha256"]
            or not lodo.adapter_hashes(adapter)
            or done["configuration_sha256"] != sat.digest(config_path)
            or Path(config["model_name_or_path"]).name != runtime["model_revision"]
            or Path(config["base_model"]).name != runtime["tokenizer_revision"]
            or int(config["seed"]) != sat.SEED
            or config.get("subsample") is not None
            or config.get("report_to") != "none"
            or steps < 1):
        raise ValueError(f"Single-source comparator is incomplete or changed: {source}")
    return {"config_sha256": sat.digest(config_path),
            "training_done_sha256": sat.digest(done_path),
            "state_sha256": sat.digest(state_path) if state_path.is_file() else None,
            "selection_sha256": sat.digest(run / "selection.json"),
            "metrics_sha256": sat.digest(run / "metrics.json"),
            "adapter_sha256": metrics["adapter_sha256"],
            "head_sha256": metrics["head_sha256"],
            "steps": steps, "save_steps": int(config["save_steps"]),
            "seed": int(config["seed"]), "n_validation": int(selection["n_validation"]),
            "threshold": float(selection["M2_threshold"])}


def export_reference(args: argparse.Namespace) -> None:
    """Audit the local mono-source runs and export only portable metadata."""
    bundle, runs, destination = (args.source_bundle.resolve(),
                                 args.single_runs.resolve(), args.reference.resolve())
    source_manifest = single.check_bundle(bundle)
    if source_manifest["gold_snapshot_sha256"] != final_data.EXPECTED_GOLD:
        raise ValueError("GOLD snapshot differs from the final RQ1 campaign")
    sources = {}
    for corpus in sat.CORPORA:
        run = runs / f"source_{corpus}"
        audit = source_run_info(bundle, runs, corpus)
        metrics_path = run / "document_metrics.csv"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        sources[corpus] = {
            "audit": audit,
            "config": json.loads((run / "config.json").read_text(encoding="utf-8")),
            "document_metrics_sha256": sat.digest(metrics_path),
            "test_sets": {target: {
                "n_documents": int(metrics["test_sets"][target]["M2"]["n_documents"]),
                "span_f1": float(metrics["test_sets"][target]["M2"]["span_f1"])}
                for target in sat.CORPORA if target != corpus},
        }
    reference = {"reference_name": REFERENCE_NAME,
                 "source_bundle_sha256": sat.digest(bundle / "manifest.json"),
                 "gold_snapshot_sha256": source_manifest["gold_snapshot_sha256"],
                 "sources": sources}
    if destination.is_file():
        if json.loads(destination.read_text(encoding="utf-8")) != reference:
            raise ValueError(f"Existing reference differs: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        final_data.json_file(destination, reference)
    print(json.dumps({"reference": str(destination), "sources": list(sources),
                      "sha256": sat.digest(destination)}, ensure_ascii=False, indent=2), flush=True)


def load_reference(bundle: Path, path: Path) -> dict:
    source_manifest = single.check_bundle(bundle)
    reference = json.loads(path.read_text(encoding="utf-8"))
    if (reference.get("reference_name") != REFERENCE_NAME
            or reference.get("source_bundle_sha256") != sat.digest(bundle / "manifest.json")
            or reference.get("gold_snapshot_sha256") != source_manifest["gold_snapshot_sha256"]
            or source_manifest["gold_snapshot_sha256"] != final_data.EXPECTED_GOLD
            or set(reference.get("sources", {})) != set(sat.CORPORA)):
        raise ValueError("Portable mono-source reference does not match the GOLD bundle")
    runtime = json.loads((bundle / "rq1_runtime.json").read_text(encoding="utf-8"))
    for corpus, source in reference["sources"].items():
        config, audit = source["config"], source["audit"]
        if (Path(config["model_name_or_path"]).name != runtime["model_revision"]
                or Path(config["base_model"]).name != runtime["tokenizer_revision"]
                or int(config["seed"]) != sat.SEED
                or config.get("subsample") is not None
                or config.get("report_to") != "none"
                or int(audit["steps"]) < 1
                or int(audit["save_steps"]) != int(config["save_steps"])
                or set(source["test_sets"]) != set(sat.CORPORA) - {corpus}):
            raise ValueError(f"Invalid mono-source reference: {corpus}")
    return reference


def prepare(args: argparse.Namespace) -> None:
    import torch

    bundle, reference_path, root = (args.source_bundle.resolve(), args.reference.resolve(),
                                    args.run_root.resolve())
    source_manifest = single.check_bundle(bundle)
    if source_manifest["gold_snapshot_sha256"] != final_data.EXPECTED_GOLD:
        raise ValueError("GOLD snapshot differs from the final RQ1 campaign")
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        check(root, bundle, reference_path)
        print(f"Prepared campaign verified: {root}", flush=True)
        return
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Unbound campaign directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    groups = {corpus: {part: sat.read_jsonl(bundle / "folds" / f"target_{corpus}" / f"{part}.jsonl")
                       for part in ("train", "val", "test")} for corpus in sat.CORPORA}
    eligible = {}
    for corpus in sat.CORPORA:
        eligible[corpus] = []
        for row in groups[corpus]["train"]:
            sat.validate_gold(row)
            chunks, _, _ = sat.trainer_chunks(row)
            if chunks is not None:
                eligible[corpus].append((row, chunks))
        if not eligible[corpus]:
            raise ValueError(f"No trainer-compatible narratives in {corpus}")
        for part in ("val", "test"):
            for row in groups[corpus][part]:
                sat.validate_gold(row)
        seen = [{key(row) for row in groups[corpus][part]}
                for part in ("train", "val", "test")]
        if len(set.union(*seen)) != sum(map(len, seen)):
            raise ValueError(f"Leaking original split in {corpus}")
    reference = load_reference(bundle, reference_path)
    comparators = reference["sources"]
    records = []
    for target in sat.CORPORA:
        for source in sat.CORPORA:
            if source == target:
                continue
            mono = [row for row, _ in eligible[source]]
            budget_chars = sum(input_chars(item) for item in eligible[source])
            selected = balanced_sample(eligible, target, source, budget_chars)
            rows = [row for row, _ in selected]
            if (len({key(row) for row in rows}) != len(rows) or
                    {row["corpus"] for row in rows} != set(sat.CORPORA) - {target} or
                    any(row["corpus"] == target for row in rows)):
                raise ValueError(f"Sampling leakage or imbalance: {target}/{source}")
            folder = root / "pairs" / pair_name(target, source)
            folder.mkdir(parents=True)
            data = {"fr": {"sentence": {sat.DATASET_NAME: {
                "meta": {"train_data": [chunks for _, chunks in selected]},
                "data": [sat.trainer_chunks(row)[0] for row in groups[source]["val"]
                         if sat.trainer_chunks(row)[0] is not None]}}}}
            path = folder / "training.pth"
            torch.save(data, path)
            val, test = groups[source]["val"], groups[target]["test"]
            mono_metrics = comparators[source]["test_sets"][target]
            if (mono_metrics["n_documents"] != len(test)
                    or comparators[source]["audit"]["n_validation"] != len(val)):
                raise ValueError(f"Comparator differs from fixed test or validation: {target}/{source}")
            records.append({"index": len(records), "target": target, "source": source,
                            "mono_eligible_narratives": len(mono),
                            "multi_eligible_narratives": len(rows),
                            "budget_characters": budget_chars,
                            "mono_training_chars": budget_chars,
                            "multi_training_chars": sum(input_chars(item) for item in selected),
                            "mono_raw_chars": sum(len(row["raw_text"]) for row in mono),
                            "multi_raw_chars": sum(len(row["raw_text"]) for row in rows),
                            "mono_training_units": sum(len(row["units"]) for row in mono),
                            "multi_training_units": sum(len(row["units"]) for row in rows),
                            "multi_by_corpus": {c: sum(row["corpus"] == c for row in rows)
                                                for c in sat.CORPORA if c != target},
                            "train_ids": [list(key(row)) for row in rows],
                            "val_ids_sha256": id_digest(val), "test_ids_sha256": id_digest(test),
                            "training_pth_sha256": sat.digest(path),
                            "single_source_test_f1": mono_metrics["span_f1"]})
    manifest = {"experiment": NAME, "sampling_seed": SEED,
                "policy": "same training characters within 1% and optimizer update count; five sources balanced by accumulated characters; identical single-source validation and target tests",
                "reference": sat.REFERENCE, "gold_snapshot_sha256": source_manifest["gold_snapshot_sha256"],
                "source_bundle_sha256": sat.digest(bundle / "manifest.json"),
                "portable_reference_sha256": sat.digest(reference_path),
                "template_sha256": sat.digest(sat.legacy.TEMPLATE),
                "script_sha256": sat.digest(Path(__file__)),
                "comparators": comparators, "pairs": records}
    final_data.json_file(manifest_path, manifest)
    print(json.dumps({"campaign": str(root), "pairs": len(records),
                      "mono_narratives": {c: len(eligible[c]) for c in sat.CORPORA}},
                     ensure_ascii=False, indent=2), flush=True)


def id_digest(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps([list(key(row)) for row in rows],
                                     ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def check(root: Path, bundle: Path, reference_path: Path) -> dict:
    import torch

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    source = single.check_bundle(bundle)
    reference = load_reference(bundle, reference_path)
    if (manifest["experiment"] != NAME or manifest["sampling_seed"] != SEED
            or manifest["gold_snapshot_sha256"] != source["gold_snapshot_sha256"]
            or manifest["source_bundle_sha256"] != sat.digest(bundle / "manifest.json")
            or manifest["portable_reference_sha256"] != sat.digest(reference_path)
            or manifest["comparators"] != reference["sources"]
            or manifest["template_sha256"] != sat.digest(sat.legacy.TEMPLATE)
            or manifest["script_sha256"] != sat.digest(Path(__file__))
            or len(manifest["pairs"]) != 30):
        raise ValueError("Equal-volume campaign binding changed")
    for pair in manifest["pairs"]:
        prepared = root / "pairs" / pair_name(pair["target"], pair["source"]) / "training.pth"
        if sat.digest(prepared) != pair["training_pth_sha256"]:
            raise ValueError(f"Prepared training set changed: {pair['index']}")
        data = torch.load(prepared, map_location="cpu", weights_only=False)[
            "fr"]["sentence"][sat.DATASET_NAME]
        chunks = data["meta"]["train_data"]
        ids = [tuple(item) for item in pair["train_ids"]]
        if (len(chunks) != pair["multi_eligible_narratives"] or len(ids) != len(chunks)
                or len(set(ids)) != len(ids)
                or {corpus for corpus, _ in ids} != set(sat.CORPORA) - {pair["target"]}
                or sum(len("\n".join(item)) for item in chunks) != pair["multi_training_chars"]
                or abs(pair["multi_training_chars"] - pair["mono_training_chars"])
                > max(1, int(pair["mono_training_chars"] * .01))):
            raise ValueError(f"Unmatched, leaking or changed training sample: {pair['index']}")
        for part, corpus, expected in (("val", pair["source"], pair["val_ids_sha256"]),
                                       ("test", pair["target"], pair["test_ids_sha256"])):
            rows = sat.read_jsonl(bundle / "folds" / f"target_{corpus}" / f"{part}.jsonl")
            if id_digest(rows) != expected:
                raise ValueError(f"Fixed {part} narratives changed: {pair['index']}")
    return manifest


def smoke(args: argparse.Namespace) -> None:
    import torch

    root, bundle, reference_path = (args.run_root.resolve(), args.source_bundle.resolve(),
                                    args.reference.resolve())
    manifest = check(root, bundle, reference_path)
    marker = root / "smoke_ok.json"
    if marker.is_file():
        if json.loads(marker.read_text(encoding="utf-8"))["manifest_sha256"] != sat.digest(root / "manifest.json"):
            raise ValueError("Smoke marker belongs to another campaign")
        lodo.adapter_hashes(sat.legacy.adapter_folder(root / "smoke" / "trainer"))
        print(f"GPU smoke already passed: {marker}", flush=True)
        return
    repo = args.wtpsplit_repo.resolve()
    revision = sat.legacy.git_revision(repo)
    sat.legacy.gpu_runtime(repo)
    pair = manifest["pairs"][0]
    source = pair["source"]
    model_dir, tokenizer_dir = sat.pinned_backbone(bundle)
    run = root / "smoke"
    run.mkdir(exist_ok=True)
    dataset = torch.load(root / "pairs" / pair_name(pair["target"], source) / "training.pth",
                         map_location="cpu", weights_only=False)
    payload = dataset["fr"]["sentence"][sat.DATASET_NAME]
    sample = {"fr": {"sentence": {sat.DATASET_NAME: {
        "meta": {"train_data": payload["meta"]["train_data"][:8]},
        "data": payload["data"][:4]}}}}
    pth = run / "training.pth"
    torch.save(sample, pth)
    config = dict(manifest["comparators"][source]["config"])
    config.update({"model_name_or_path": str(model_dir), "base_model": str(tokenizer_dir),
                   "text_path": str(pth), "output_dir": str(run / "trainer"),
                   "max_steps": 2, "save_steps": 1, "save_total_limit": None,
                   "num_train_epochs": 1, "per_device_train_batch_size": 2,
                   "per_device_eval_batch_size": 2, "gradient_accumulation_steps": 1})
    final = sat.legacy.adapter_folder(run / "trainer")
    if not lodo.checkpoint.complete_adapter(final):
        # The official trainer saves its own adapter at the end. It becomes the
        # shape/config template for exporting this campaign's checkpoints.
        sat.legacy.run_trainer(repo, config, run / "config.json")
    lodo.adapter_hashes(final)
    lodo.harvest_checkpoints(run, run / "trainer", final)
    row = sat.read_jsonl(bundle / "folds" / f"target_{source}" / "val.jsonl")[0]
    checkpoint_adapter = run / "checkpoint_adapters" / "checkpoint-1"
    for adapter in (checkpoint_adapter, final):
        lodo.adapter_hashes(adapter)
        model = sat.load_pinned_sat(model_dir, tokenizer_dir, adapter)
        spans = sat.legacy.pieces_to_spans(
            row["raw_text"], model.split(row["raw_text"], threshold=sat.DEFAULT))
        sat.score(row, spans)
        del model
    final_data.json_file(marker, {"manifest_sha256": sat.digest(root / "manifest.json"),
                                  "wtpsplit_revision": revision, "adapter": lodo.adapter_hashes(final)})
    trainer = run / "trainer"
    for folder in trainer.glob("checkpoint-*"):
        if folder.is_dir() and folder.name.removeprefix("checkpoint-").isdigit():
            lodo.remove_checkpoint(trainer, folder)
    print(f"Equal-volume GPU smoke passed: {marker}", flush=True)


def fit(args: argparse.Namespace) -> None:
    import torch

    root, bundle, reference_path = (args.run_root.resolve(), args.source_bundle.resolve(),
                                    args.reference.resolve())
    manifest = check(root, bundle, reference_path)
    if not 0 <= args.task_index < 30:
        raise ValueError("Array task must be in 0..29")
    if json.loads((root / "smoke_ok.json").read_text(encoding="utf-8"))["manifest_sha256"] != sat.digest(root / "manifest.json"):
        raise ValueError("GPU smoke does not match prepared campaign")
    pair = manifest["pairs"][args.task_index]
    target, source = pair["target"], pair["source"]
    run = root / "runs" / pair_name(target, source)
    run.mkdir(parents=True, exist_ok=True)
    binding = {"manifest_sha256": sat.digest(root / "manifest.json"), "pair_index": args.task_index}
    bind_path = run / "binding.json"
    if bind_path.exists():
        if json.loads(bind_path.read_text(encoding="utf-8")) != binding:
            raise ValueError(f"Run belongs to different data: {run}")
    elif any(run.iterdir()):
        raise ValueError(f"Unbound run: {run}")
    else:
        final_data.json_file(bind_path, binding)
    metrics_path = run / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (metrics["binding"] != binding or metrics["n_test"] != final_data.EXPECTED_COUNTS[target][1]
                or metrics["selected_weights"] != lodo.adapter_hashes(run / "selected_adapter")):
            raise ValueError("Existing pair result changed")
        cleanup(run, lodo.adapter_hashes(run / "selected_adapter"))
        print(f"Already complete: {pair_name(target, source)}", flush=True)
        return
    model_dir, tokenizer_dir = sat.pinned_backbone(bundle)
    repo = args.wtpsplit_repo.resolve()
    revision = sat.legacy.git_revision(repo)
    config = dict(manifest["comparators"][source]["config"])
    config.update({"model_name_or_path": str(model_dir), "base_model": str(tokenizer_dir),
                   "text_path": str(root / "pairs" / pair_name(target, source) / "training.pth"),
                   "output_dir": str(run / "trainer"),
                   "max_steps": manifest["comparators"][source]["audit"]["steps"],
                   "save_steps": manifest["comparators"][source]["audit"]["save_steps"],
                   "save_total_limit": None, "load_best_model_at_end": False})
    if config["subsample"] is not None or config["report_to"] != "none":
        raise ValueError("Unsafe trainer config")
    trainer = run / "trainer"
    final = sat.legacy.adapter_folder(trainer)
    template = sat.legacy.adapter_folder(root / "smoke" / "trainer")
    lodo.adapter_hashes(template)
    if not lodo.checkpoint.complete_adapter(final):
        resume = lodo.checkpoint.latest_resumable_checkpoint(trainer)
        if resume:
            config["resume_from_checkpoint"] = str(resume)
        lodo.run_trainer_with_harvest(repo, config, run / "config.json", run, template)
    lodo.adapter_hashes(final)
    lodo.harvest_checkpoints(run, trainer, template)
    state = json.loads((trainer / "trainer_state.json").read_text(encoding="utf-8"))
    if int(state["global_step"]) != manifest["comparators"][source]["audit"]["steps"]:
        raise ValueError("Multi-source optimizer updates differ from mono-source")
    sat.legacy.export_loss_history(trainer / "trainer_state.json", run / "loss_history.csv")
    candidates = []
    for path in sorted((run / "checkpoint_adapters").glob("checkpoint-*")):
        if path.is_dir():
            meta = json.loads((path / "candidate.json").read_text(encoding="utf-8"))
            if meta["step"] < state["global_step"]:
                candidates.append((path.name, meta["step"], meta["epoch"], path))
    candidates.append(("final", int(state["global_step"]), float(state["epoch"]), final))
    val = sat.read_jsonl(bundle / "folds" / f"target_{source}" / "val.jsonl")
    curves, adapters = [], {}
    for name, step, epoch, adapter in candidates:
        cache = run / "validation_cache" / f"{name}.sqlite3"
        single.cache_binding(cache, bundle, adapter)
        with closing(sat._probability_connection(cache, create=True)) as db:
            if db.execute("SELECT COUNT(*) FROM probabilities").fetchone()[0] < len(val):
                model = sat.load_pinned_sat(model_dir, tokenizer_dir, adapter)
                single.infer_fixed_batches(model, db, val, f"Validation {target}/{source}/{name}")
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if db.execute("SELECT COUNT(*) FROM probabilities").fetchone()[0] != len(val):
                raise ValueError("Incomplete source validation cache")
            curves.extend(single.validation_curve(val, db, name, step, epoch))
        adapters[name] = adapter
    sat.write_csv(run / "checkpoint_validation.csv", curves, tuple(curves[0]))
    chosen = single.best_candidate(curves)
    chosen_adapter = adapters[chosen["candidate"]]
    selected = run / "selected_adapter"
    lodo.persist_selected_adapter(chosen_adapter, selected, lodo.adapter_hashes(chosen_adapter))
    selection = {"criterion": "single-source validation micro exact-span F1",
                 "source": source, "target": target, "n_validation": len(val),
                 "candidate": chosen["candidate"], "step": chosen["step"],
                 "threshold": chosen["threshold"], "validation_exact_f1": chosen["exact_f1"],
                 "selected_weights": lodo.adapter_hashes(selected),
                 "curve_sha256": sat.digest(run / "checkpoint_validation.csv")}
    if (run / "selection.json").is_file():
        if json.loads((run / "selection.json").read_text(encoding="utf-8")) != selection:
            raise ValueError("Source-only selection changed on resume")
    else:
        final_data.json_file(run / "selection.json", selection)
    test = sat.read_jsonl(bundle / "folds" / f"target_{target}" / "test.jsonl")
    cache = run / "target_probabilities.sqlite3"
    single.cache_binding(cache, bundle, selected)
    with closing(sat._probability_connection(cache, create=True)) as db:
        if db.execute("SELECT COUNT(*) FROM probabilities").fetchone()[0] < len(test):
            model = sat.load_pinned_sat(model_dir, tokenizer_dir, selected)
            single.infer_fixed_batches(model, db, test, f"Test {target}/{source}")
            sat.verify_native(model, db, test, selection["threshold"])
            del model
        if db.execute("SELECT COUNT(*) FROM probabilities").fetchone()[0] != len(test):
            raise ValueError("Incomplete target test cache")
        output = run / "predictions.jsonl"
        temp = output.with_suffix(".jsonl.tmp")
        detail = []
        with temp.open("w", encoding="utf-8") as stream:
            for row in test:
                spans = sat.legacy.spans_from_probabilities(
                    row["raw_text"], sat.get_probs(db, row), selection["threshold"])
                if any(not (0 <= a < b <= len(row["raw_text"])) for a, b in spans) or any(
                        spans[i][1] > spans[i + 1][0] for i in range(len(spans) - 1)):
                    raise ValueError(f"Invalid or overlapping output spans: {target}/{row['document_id']}")
                detail.append({"target": target, "source": source,
                               "document_id": row["document_id"],
                               **sat.score(row, spans, details=True)})
                stream.write(json.dumps({"target": target, "source": source,
                                         "document_id": row["document_id"],
                                         "segments": [{"start": a, "end": b, "text": row["raw_text"][a:b]}
                                                      for a, b in spans]}, ensure_ascii=False) + "\n")
        temp.replace(output)
    fields = ("target", "source", "document_id", *sat.SCORE_FIELDS,
              *(f"role_{role}_{kind}" for role in sat.ROLES for kind in ("tp", "ref")),
              "fusion", "fragmented", "gold_units", "pred_segments", "supported_pred_segments")
    sat.write_csv(run / "document_metrics.csv", detail, fields)
    totals = {field: sum(int(row[field]) for row in detail) for field in sat.SCORE_FIELDS}
    exact = sat.prf(totals["span_tp"], totals["span_pred"], totals["span_ref"])[2]
    final_data.json_file(metrics_path, {"binding": binding, "n_test": len(test),
                                        "target": target, "source": source,
                                        "steps": int(state["global_step"]),
                                        "selected_weights": selection["selected_weights"],
                                        "selection_sha256": sat.digest(run / "selection.json"),
                                        "predictions_sha256": sat.digest(output),
                                        "document_metrics_sha256": sat.digest(run / "document_metrics.csv"),
                                        "exact_f1": exact, "wtpsplit_revision": revision})
    # Full optimizer states are removed only after predictions and selected weights exist.
    if lodo.adapter_hashes(selected) != selection["selected_weights"]:
        raise ValueError("Selected adapter changed before cleanup")
    cleanup(run, selection["selected_weights"])
    print(f"Completed {source} -> {target}: archived mono={pair['single_source_test_f1']:.4f}, "
          f"equal-volume multi={exact:.4f}", flush=True)


def cleanup(run: Path, expected_weights: dict) -> None:
    if lodo.adapter_hashes(run / "selected_adapter") != expected_weights:
        raise ValueError("Selected adapter changed before cleanup")
    trainer = run / "trainer"
    final = sat.legacy.adapter_folder(trainer)
    for folder in trainer.glob("checkpoint-*"):
        if folder.is_dir() and folder.name.removeprefix("checkpoint-").isdigit():
            lodo.remove_checkpoint(trainer, folder)
    exported = run / "checkpoint_adapters"
    if exported.exists() and exported.resolve().parent == run.resolve():
        shutil.rmtree(exported)
    if final.exists() and final.resolve().is_relative_to(trainer.resolve()):
        shutil.rmtree(final)


def summary(args: argparse.Namespace) -> None:
    """Cluster-side completion audit; comparison is deliberately local."""
    root, bundle, reference_path = (args.run_root.resolve(), args.source_bundle.resolve(),
                                    args.reference.resolve())
    manifest = check(root, bundle, reference_path)
    results = []
    for pair in manifest["pairs"]:
        target, source = pair["target"], pair["source"]
        run = root / "runs" / pair_name(target, source)
        result = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
        tests = sat.read_jsonl(bundle / "folds" / f"target_{target}" / "test.jsonl")
        with (run / "document_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
            detail = list(csv.DictReader(stream))
        if (result["binding"] != {"manifest_sha256": sat.digest(root / "manifest.json"),
                                  "pair_index": pair["index"]}
                or result["selected_weights"] != lodo.adapter_hashes(run / "selected_adapter")
                or result["selection_sha256"] != sat.digest(run / "selection.json")
                or result["document_metrics_sha256"] != sat.digest(run / "document_metrics.csv")
                or result["predictions_sha256"] != sat.digest(run / "predictions.jsonl")
                or selection["n_validation"] != manifest["comparators"][source]["audit"]["n_validation"]
                or result["steps"] != manifest["comparators"][source]["audit"]["steps"]
                or result["n_test"] != len(tests)
                or [row["document_id"] for row in detail] != [row["document_id"] for row in tests]):
            raise ValueError(f"Incomplete or mismatched multisource run: {source} -> {target}")
        counts = [sum(int(row[field]) for row in detail)
                  for field in ("span_tp", "span_pred", "span_ref")]
        score = f1(counts)
        if abs(score - result["exact_f1"]) > 1e-10:
            raise ValueError(f"Multisource exact F1 changed: {source} -> {target}")
        results.append({"source": source, "target": target, "n_test": len(tests),
                        "exact_f1": score, "selected_threshold": selection["threshold"],
                        "selected_step": selection["step"]})
    sat.write_csv(root / "multi_only_results.csv", results, tuple(results[0]))
    final_data.json_file(root / "multi_only_complete.json",
                         {"campaign": NAME, "n_pairs": len(results),
                          "manifest_sha256": sat.digest(root / "manifest.json"),
                          "multi_only_results_sha256": sat.digest(root / "multi_only_results.csv"),
                          "comparison": "Rapatrier ce dossier et lancer compare-local avec les modèles mono-source conservés sur le PC."})
    print(f"All {len(results)} multisource runs verified; local comparison is ready.", flush=True)


def compare_local(args: argparse.Namespace) -> None:
    import numpy as np

    root, bundle, runs, reference_path = (args.run_root.resolve(), args.source_bundle.resolve(),
                                          args.single_runs.resolve(), args.reference.resolve())
    summary(argparse.Namespace(run_root=root, source_bundle=bundle, reference=reference_path))
    manifest = check(root, bundle, reference_path)
    results, paired = [], {}
    old = {}
    for source in sat.CORPORA:
        comparator = manifest["comparators"][source]
        if source_run_info(bundle, runs, source) != comparator["audit"]:
            raise ValueError(f"Local mono-source run differs from the exported reference: {source}")
        path = runs / f"source_{source}" / "document_metrics.csv"
        if sat.digest(path) != comparator["document_metrics_sha256"]:
            raise ValueError(f"Local mono-source document metrics changed: {source}")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            old[source] = {(row["target"], row["document_id"]): row
                           for row in csv.DictReader(stream) if row["model"] == "M2"}
    for pair in manifest["pairs"]:
        target, source = pair["target"], pair["source"]
        run = root / "runs" / pair_name(target, source)
        result = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
        if (result["binding"] != {"manifest_sha256": sat.digest(root / "manifest.json"),
                                 "pair_index": pair["index"]}
                or result["selection_sha256"] != sat.digest(run / "selection.json")
                or result["selected_weights"] != lodo.adapter_hashes(run / "selected_adapter")
                or result["document_metrics_sha256"] != sat.digest(run / "document_metrics.csv")
                or result["predictions_sha256"] != sat.digest(run / "predictions.jsonl")
                or selection["n_validation"] != manifest["comparators"][source]["audit"]["n_validation"]
                or result["steps"] != manifest["comparators"][source]["audit"]["steps"]):
            raise ValueError(f"Missing or mismatched pair: {source} -> {target}")
        tests = sat.read_jsonl(bundle / "folds" / f"target_{target}" / "test.jsonl")
        with (run / "document_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
            new = {row["document_id"]: row for row in csv.DictReader(stream)}
        if len(new) != len(tests) or set(new) != {row["document_id"] for row in tests}:
            raise ValueError(f"Target test IDs changed: {source} -> {target}")
        ids = [row["document_id"] for row in tests]
        previous = [old[source][target, doc] for doc in ids]
        current = [new[doc] for doc in ids]
        if any(int(a["span_ref"]) != int(b["span_ref"]) for a, b in zip(previous, current)):
            raise ValueError("GOLD span counts differ between paired methods")
        arrays = {label: np.array([[int(row[field]) for field in ("span_tp", "span_pred", "span_ref")]
                                   for row in rows], dtype=np.int64)
                  for label, rows in (("mono", previous), ("multi", current))}
        paired[target, source] = arrays
        scores = {label: f1(array.sum(axis=0)) for label, array in arrays.items()}
        if abs(scores["mono"] - pair["single_source_test_f1"]) > 1e-10:
            raise ValueError("Single-source score does not reproduce the archived comparator")
        results.append({"target": target, "source": source, "n_test": len(ids),
                        "mono_narratives": pair["mono_eligible_narratives"],
                        "multi_narratives": pair["multi_eligible_narratives"],
                        "optimizer_steps": result["steps"],
                        "mono_chars": pair["mono_training_chars"],
                        "multi_chars": pair["multi_training_chars"],
                        "mono_units": pair["mono_training_units"],
                        "multi_units": pair["multi_training_units"],
                        "mono_f1": scores["mono"], "multi_f1": scores["multi"],
                        "delta_multi_minus_mono": scores["multi"] - scores["mono"]})
    sat.write_csv(root / "pair_results.csv", results, tuple(results[0]))
    targets = []
    for target in sat.CORPORA:
        group = [row for row in results if row["target"] == target]
        targets.append({"target": target, "n_test": group[0]["n_test"],
                        "mono_mean_f1": sum(row["mono_f1"] for row in group) / 5,
                        "multi_equal_volume_mean_f1": sum(row["multi_f1"] for row in group) / 5,
                        "mean_paired_delta": sum(row["delta_multi_minus_mono"] for row in group) / 5,
                        "multi_wins": sum(row["delta_multi_minus_mono"] > 0 for row in group)})
    rng = np.random.default_rng(SEED)
    samples = []
    target_samples = {target: [] for target in sat.CORPORA}
    for _ in range(BOOTSTRAPS):
        differences = []
        for target in sat.CORPORA:
            first = paired[target, next(source for source in sat.CORPORA if source != target)]["mono"]
            draw = rng.integers(0, len(first), len(first))
            within = [f1(paired[target, source]["multi"][draw].sum(axis=0)) -
                      f1(paired[target, source]["mono"][draw].sum(axis=0))
                      for source in sat.CORPORA if source != target]
            differences.extend(within)
            target_samples[target].append(sum(within) / len(within))
        samples.append(sum(differences) / len(differences))
    for row in targets:
        draws = target_samples[row["target"]]
        row["ci95_low"] = float(np.quantile(draws, .025))
        row["ci95_high"] = float(np.quantile(draws, .975))
    sat.write_csv(root / "target_summary.csv", targets, tuple(targets[0]))
    matrix = []
    for source in sat.CORPORA:
        matrix.append({"source": source, **{
            target: next((row["delta_multi_minus_mono"] for row in results
                          if row["target"] == target and row["source"] == source), "N/A")
            for target in sat.CORPORA}})
    sat.write_csv(root / "transfer_matrix_delta.csv", matrix, tuple(matrix[0]))
    transfer_svg(root / "transfer_matrix_delta.svg", matrix)
    point = sum(row["delta_multi_minus_mono"] for row in results) / len(results)
    report = {"campaign": NAME, "comparison": "30 paired source-target transfers; same target test, source validation, training characters within 1%, and optimizer updates",
              "n_pairs": len(results), "n_targets": 6, "n_seeds": 1,
              "macro_mono_exact_f1": sum(row["mono_mean_f1"] for row in targets) / 6,
              "macro_equal_volume_multi_exact_f1": sum(row["multi_equal_volume_mean_f1"] for row in targets) / 6,
              "macro_paired_delta_multi_minus_mono": point,
              "bootstrap_95_ci": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))],
              "bootstrap_repetitions": BOOTSTRAPS,
              "multi_wins_of_30": sum(row["delta_multi_minus_mono"] > 0 for row in results),
              "target_wins_of_6": sum(row["mean_paired_delta"] > 0 for row in targets),
              "limits": "Equal character and update budgets do not force equal token, narrative or factual-unit counts; these are reported. Bootstrap conditions on fitted models and one seed."}
    final_data.json_file(root / "RESULTS.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def f1(counts) -> float:
    return 2 * int(counts[0]) / int(counts[1] + counts[2]) if counts[1] + counts[2] else 0.0


def transfer_svg(path: Path, matrix: list[dict]) -> None:
    """Small dependency-free heatmap: paired multi minus mono Exact-span F1."""
    from html import escape

    cell, left, top = 100, 180, 105
    width, height = left + 6 * cell + 30, top + 6 * 55 + 65
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="20" y="28" font-family="Arial" font-size="17" font-weight="bold">'
             'Multisource minus monosource, equal training characters</text>',
             '<text x="20" y="48" font-family="Arial" font-size="12">'
             'Exact-span F1; rows: single source, columns: held-out target</text>']
    for j, target in enumerate(sat.CORPORA):
        parts.append(f'<text x="{left + j * cell + cell / 2}" y="{top - 10}" '
                     f'text-anchor="middle" font-family="Arial" font-size="11">'
                     f'{escape(target)}</text>')
    for i, row in enumerate(matrix):
        y = top + i * 55
        parts.append(f'<text x="{left - 8}" y="{y + 30}" text-anchor="end" '
                     f'font-family="Arial" font-size="12">{escape(row["source"])}</text>')
        for j, target in enumerate(sat.CORPORA):
            value = row[target]
            if value == "N/A":
                fill, label = "#eeeeee", "N/A"
            else:
                strength = min(abs(value) / .20, 1.0)
                pale = int(245 - 95 * strength)
                fill = (f'rgb({pale},225,{pale})' if value >= 0 else
                        f'rgb(245,{pale},{pale})')
                label = f'{value:+.3f}'
            x = left + j * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell - 3}" height="52" '
                         f'fill="{fill}" stroke="#cccccc"/>')
            parts.append(f'<text x="{x + (cell - 3) / 2}" y="{y + 31}" '
                         f'text-anchor="middle" font-family="Arial" font-size="13">'
                         f'{label}</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, func in (("export-reference", export_reference),
                          ("prepare", prepare), ("smoke", smoke), ("fit", fit),
                          ("summary", summary), ("compare-local", compare_local)):
        sub = commands.add_parser(command)
        sub.add_argument("--source-bundle", type=Path, required=True)
        sub.add_argument("--reference", type=Path, required=True)
        if command in ("export-reference", "compare-local"):
            sub.add_argument("--single-runs", type=Path, required=True)
        if command != "export-reference":
            sub.add_argument("--run-root", type=Path, required=True)
        if command in ("smoke", "fit"):
            sub.add_argument("--wtpsplit-repo", type=Path, required=True)
        if command == "fit":
            sub.add_argument("--task-index", type=int, required=True)
        sub.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
