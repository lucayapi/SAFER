#!/usr/bin/env python3
"""Single-source SaT-LoRA experiment on the confirmed narrative_v3 reference.

Prepare six grouped 70/15/15 corpus splits from the existing private LODO
bundle. Each GPU task fits one corpus, selects a checkpoint and threshold on
that corpus's validation narratives, then scores the same six held-out test
sets. Test labels are checked for integrity during preparation but are only
used for scoring after model selection has been written to disk.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np

import sat_lodo_gold_v3 as base
import sat_lodo_checkpoint_select as checkpoints


ROOT = Path(__file__).resolve().parents[2]
SOURCE_BUNDLE = base.BUNDLE
BUNDLE_NAME = "sat12l_sm_gold_v3_indomain_v1"
BUNDLE = ROOT / "annotation" / "lodo_sat_transfer" / BUNDLE_NAME
RUN_NAME = BUNDLE_NAME
SAVE_STEPS = 200
SMALL_CORPUS_SAVE_STEPS = 50
SPLIT_SEED = base.SEED + 41
SPLIT_NAMES = ("train", "val", "test")
REFERENCE = base.REFERENCE


def atomic_json(path: Path, value: object) -> None:
    checkpoints.write_json_atomic(path, value)


def row_key(row: dict) -> tuple[str, str]:
    return row["corpus"], row["document_id"]


def grouped_split(rows: list[dict], corpus: str) -> dict[str, list[dict]]:
    """Keep exact duplicate narratives together; independent of CSV row order."""
    if not rows or any(row["corpus"] != corpus for row in rows):
        raise ValueError(f"Invalid corpus group: {corpus}")
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        grouped[row["raw_text"]].append(row)
    ordered = sorted(grouped.items(), key=lambda pair: (
        hashlib.sha256(f"{SPLIT_SEED}\0{corpus}\0{pair[0]}".encode("utf-8")).hexdigest(),
        pair[0]))
    target_train = round(0.70 * len(rows))
    target_val_end = round(0.85 * len(rows))
    split = {name: [] for name in SPLIT_NAMES}
    for _, duplicates in ordered:
        already = len(split["train"]) + len(split["val"])
        name = ("train" if len(split["train"]) < target_train else
                "val" if already < target_val_end else "test")
        split[name].extend(sorted(duplicates, key=lambda row: row["document_id"]))
    if any(not split[name] for name in SPLIT_NAMES):
        raise ValueError(f"Empty corpus partition: {corpus}")
    return split


def checkpoint_interval(corpus: str) -> int:
    """Ensure the smallest corpus has at least one intermediate candidate."""
    return SMALL_CORPUS_SAVE_STEPS if corpus == "caou_chimie_plas" else SAVE_STEPS


def source_rows(source: Path) -> tuple[dict, dict[str, list[dict]]]:
    manifest = base.check_bundle(source)
    groups = {}
    seen: set[tuple[str, str]] = set()
    text_owner: dict[str, str] = {}
    for corpus in base.CORPORA:
        path = source / "folds" / f"target_{corpus}" / "test.jsonl"
        group = base.read_jsonl(path)
        for row in group:
            base.validate_gold(row)
            if row["corpus"] != corpus or row_key(row) in seen:
                raise ValueError(f"Duplicate or misplaced gold record: {corpus}")
            previous = text_owner.setdefault(row["raw_text"], corpus)
            if previous != corpus:
                raise ValueError(f"Identical narrative appears in two corpora: {previous}/{corpus}")
            seen.add(row_key(row))
        groups[corpus] = group
    if len(seen) != sum(manifest["source_report"][c]["gold_documents"]
                        for c in base.CORPORA):
        raise ValueError("Source bundle does not cover the full reference")
    return manifest, groups


def prepare(args: argparse.Namespace) -> None:
    import torch

    source, out = Path(args.source_bundle).resolve(), Path(args.out).resolve()
    source_manifest, groups = source_rows(source)
    if out.exists() or out.with_name(out.name + ".preparing").exists():
        raise FileExistsError(f"A prepared or unfinished bundle already exists: {out}")
    staging = out.with_name(out.name + ".preparing")
    staging.mkdir(parents=True)
    files: dict[str, str] = {}
    report = {}
    keys: set[tuple[str, str]] = set()
    for corpus in base.CORPORA:
        split = grouped_split(groups[corpus], corpus)
        folder = staging / "folds" / f"target_{corpus}"
        folder.mkdir(parents=True)
        trainer = {}
        excluded = {}
        for name in SPLIT_NAMES:
            path = folder / f"{name}.jsonl"
            base.legacy.write_jsonl(path, split[name])
            files[path.relative_to(staging).as_posix()] = base.digest(path)
            keys.update(map(row_key, split[name]))
            if name == "test":
                continue
            examples = []
            reasons = collections.Counter()
            for row in split[name]:
                chunks, reason, _ = base.trainer_chunks(row)
                if chunks is None:
                    reasons[reason] += 1
                else:
                    examples.append(chunks)
            if not examples:
                raise ValueError(f"No trainer-compatible narratives: {corpus}/{name}")
            trainer[name] = examples
            excluded[name] = dict(reasons)
        ids = {name: [row["document_id"] for row in split[name]]
               for name in SPLIT_NAMES}
        path = folder / "split_ids.json"
        atomic_json(path, ids)
        files[path.relative_to(staging).as_posix()] = base.digest(path)
        path = folder / "training.pth"
        torch.save({"fr": {"sentence": {base.DATASET_NAME: {
            "meta": {"train_data": trainer["train"]},
            "data": trainer["val"]}}}}, path)
        files[path.relative_to(staging).as_posix()] = base.digest(path)
        report[corpus] = {"total": len(groups[corpus]),
                          "gold_units": sum(len(row["units"]) for row in groups[corpus]),
                          "split": {name: len(split[name]) for name in SPLIT_NAMES},
                          "trainer_usable": {name: len(trainer[name]) for name in ("train", "val")},
                          "trainer_excluded": excluded}
        print(f"{corpus}: {report[corpus]['split']} usable={report[corpus]['trainer_usable']}",
              flush=True)
    base.copy_base_probabilities(source / "base_probabilities.sqlite3",
                                staging / "base_probabilities.sqlite3", keys)
    path = staging / "base_probabilities.sqlite3"
    files[path.relative_to(staging).as_posix()] = base.digest(path)
    path = staging / "rq1_runtime.json"
    path.write_bytes((source / "rq1_runtime.json").read_bytes())
    files[path.relative_to(staging).as_posix()] = base.digest(path)
    manifest = {"experiment": "single_source_indomain_and_transfer_v1",
                "reference": REFERENCE, "model": base.MODEL, "seed": SPLIT_SEED,
                "fold_order": list(base.CORPORA),
                "split_policy": "70/15/15 per corpus, grouped by identical raw narrative",
                "selection": "source validation exact-span F1, checkpoint and threshold",
                "thresholds": list(base.THRESHOLDS),
                "source_bundle_manifest_sha256": base.digest(source / "manifest.json"),
                "gold_snapshot_sha256": source_manifest["gold_snapshot_sha256"],
                "template_sha256": base.digest(base.legacy.TEMPLATE),
                "report": report, "files": files}
    atomic_json(staging / "manifest.json", manifest)
    atomic_json(staging / "complete.json", {"manifest_sha256": base.digest(staging / "manifest.json")})
    check_bundle(staging)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(out)
    print(json.dumps({"bundle": str(out), "documents": len(keys), "report": report},
                     ensure_ascii=False, indent=2), flush=True)


def check_bundle(bundle: Path) -> dict:
    import torch

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    complete = json.loads((bundle / "complete.json").read_text(encoding="utf-8"))
    if (complete["manifest_sha256"] != base.digest(manifest_path)
            or manifest["reference"] != REFERENCE or manifest["model"] != base.MODEL
            or manifest["fold_order"] != list(base.CORPORA)
            or manifest["thresholds"] != list(base.THRESHOLDS)
            or manifest["template_sha256"] != base.digest(base.legacy.TEMPLATE)):
        raise ValueError("In-domain bundle metadata or template changed")
    for relative, expected in manifest["files"].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle.resolve()) or base.digest(path) != expected:
            raise ValueError(f"Missing or changed private bundle file: {relative}")
    all_keys = set()
    for corpus in base.CORPORA:
        folder = bundle / "folds" / f"target_{corpus}"
        split = {name: base.read_jsonl(folder / f"{name}.jsonl") for name in SPLIT_NAMES}
        split_keys = {name: set(map(row_key, rows)) for name, rows in split.items()}
        if (any(not split[name] or len(split[name]) != len(split_keys[name])
                for name in SPLIT_NAMES)
                or any(split_keys[a] & split_keys[b]
                       for a, b in (("train", "val"), ("train", "test"), ("val", "test")))):
            raise ValueError(f"Empty, duplicate, or overlapping split: {corpus}")
        ids = json.loads((folder / "split_ids.json").read_text(encoding="utf-8"))
        texts = {}
        data = torch.load(folder / "training.pth", map_location="cpu", weights_only=False)[
            "fr"]["sentence"][base.DATASET_NAME]
        for name, rows in split.items():
            if ids[name] != [row["document_id"] for row in rows]:
                raise ValueError(f"Split ID order changed: {corpus}/{name}")
            for row in rows:
                base.validate_gold(row)
                if row["corpus"] != corpus or row["raw_text"] in texts and texts[row["raw_text"]] != name:
                    raise ValueError(f"Corpus mismatch or duplicate text leakage: {corpus}/{name}")
                texts[row["raw_text"]] = name
            if name != "test":
                examples = [chunks for row in rows if (chunks := base.trainer_chunks(row)[0]) is not None]
                saved = data["meta"]["train_data"] if name == "train" else data["data"]
                if examples != saved:
                    raise ValueError(f"Trainer copy changed: {corpus}/{name}")
            all_keys.update(split_keys[name])
        if any(len(split[name]) != manifest["report"][corpus]["split"][name]
               for name in SPLIT_NAMES):
            raise ValueError(f"Manifest counts changed: {corpus}")
    if len(all_keys) != sum(manifest["report"][c]["total"] for c in base.CORPORA):
        raise ValueError("The six splits do not cover the gold reference once")
    with closing(sqlite3.connect(bundle / "base_probabilities.sqlite3")) as db:
        cache_keys = set(db.execute("SELECT corpus, document_id FROM probabilities"))
    if cache_keys != all_keys:
        raise ValueError("Unadapted SaT cache and fixed tests differ")
    return manifest


def check_command(args: argparse.Namespace) -> None:
    manifest = check_bundle(Path(args.bundle).resolve())
    print(json.dumps({"ready": True, "bundle": str(Path(args.bundle).resolve()),
                      "corpora": manifest["report"]}, ensure_ascii=False, indent=2))


def run_signature(bundle: Path) -> dict:
    return {"bundle_manifest_sha256": base.digest(bundle / "manifest.json"),
            "script_sha256": base.digest(Path(__file__)),
            "base_script_sha256": base.digest(Path(base.__file__)),
            "checkpoint_helper_sha256": base.digest(Path(checkpoints.__file__)),
            "resume_helper_sha256": base.digest(checkpoints.RESUME_SCRIPT),
            "template_sha256": base.digest(base.legacy.TEMPLATE)}


def bind_run(run: Path, bundle: Path) -> None:
    binding = run / "binding.json"
    expected = run_signature(bundle)
    if binding.is_file():
        if json.loads(binding.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Run directory belongs to different data or code: {run}")
    elif run.exists() and any(run.iterdir()):
        raise ValueError(f"Unbound run directory: {run}")
    else:
        run.mkdir(parents=True, exist_ok=True)
        atomic_json(binding, expected)


def candidates(trainer: Path, save_steps: int) -> list[dict]:
    state = json.loads((trainer / "trainer_state.json").read_text(encoding="utf-8"))
    final_step = int(state["global_step"])
    if final_step < 1:
        raise ValueError("Trainer completed without an optimizer step")
    expected = range(save_steps, final_step + 1, save_steps)
    rows = []
    for step in expected:
        folder = trainer / f"checkpoint-{step}"
        if not (folder / "pytorch_model.bin").is_file():
            raise FileNotFoundError(f"Required checkpoint missing: {folder}")
        if step == final_step:
            continue
        snapshot = json.loads((folder / "trainer_state.json").read_text(encoding="utf-8"))
        rows.append({"candidate": f"checkpoint-{step}", "step": step,
                     "epoch": float(snapshot["epoch"]), "checkpoint": folder})
    rows.append({"candidate": "final", "step": final_step,
                 "epoch": float(state["epoch"]), "checkpoint": None})
    if len(rows) < 2:
        raise ValueError("No intermediate checkpoint to compare with the final adapter")
    return rows


def cache_binding(cache: Path, bundle: Path, adapter: Path) -> None:
    marker = cache.with_suffix(".binding.json")
    expected = {"bundle_manifest_sha256": base.digest(bundle / "manifest.json"),
                "adapter_sha256": base.digest(adapter / "pytorch_adapter.bin"),
                "head_sha256": base.digest(adapter / "pytorch_model_head.bin"),
                "inference_stride": 64, "batch_size": 16}
    if marker.is_file():
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Probability cache belongs to other weights or data: {cache}")
    elif cache.exists():
        raise ValueError(f"Unbound probability cache: {cache}")
    else:
        marker.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(marker, expected)


def infer_fixed_batches(sat, db: sqlite3.Connection, rows: list[dict], label: str) -> None:
    """A resumed run must not change SaT's batch-dependent inference windows."""
    import zlib
    from tqdm import tqdm

    present = set(db.execute("SELECT corpus,document_id FROM probabilities"))
    for start in tqdm(range(0, len(rows), 16), desc=label, unit="lot"):
        batch = rows[start:start + 16]
        keys = {row_key(row) for row in batch}
        if keys <= present:
            continue
        calculated = sat.predict_proba([row["raw_text"] for row in batch], stride=64)
        records = []
        for row, value in zip(batch, calculated, strict=True):
            probabilities = np.asarray(value, dtype=np.float32)
            if len(probabilities) != len(row["raw_text"]) or not np.isfinite(probabilities).all():
                raise ValueError(f"Invalid model probabilities: {row_key(row)}")
            records.append((row["corpus"], row["document_id"], len(probabilities),
                            zlib.compress(probabilities.tobytes())))
        with db:
            db.executemany("INSERT OR REPLACE INTO probabilities VALUES (?,?,?,?)", records)
        present.update(keys)
    if present != {row_key(row) for row in rows}:
        raise ValueError("Probability cache has missing or extra documents")


def validation_curve(rows: list[dict], db: sqlite3.Connection,
                     candidate: str, step: int, epoch: float) -> list[dict]:
    if len({row["corpus"] for row in rows}) != 1:
        raise ValueError("In-domain validation must come from one corpus")
    totals = {threshold: collections.Counter() for threshold in base.THRESHOLDS}
    for row in rows:
        probabilities = base.get_probs(db, row)
        for threshold in base.THRESHOLDS:
            spans = base.legacy.spans_from_probabilities(row["raw_text"], probabilities, threshold)
            totals[threshold].update(base.score(row, spans))
    curve = []
    for threshold, counts in totals.items():
        if not counts["span_ref"]:
            raise ValueError("Validation has no factual gold spans")
        precision, recall, f1 = base.prf(counts["span_tp"], counts["span_pred"], counts["span_ref"])
        curve.append({"candidate": candidate, "step": step, "epoch": epoch,
                      "threshold": threshold, "exact_precision": precision,
                      "exact_recall": recall, "exact_f1": f1,
                      "span_tp": counts["span_tp"], "span_pred": counts["span_pred"],
                      "span_ref": counts["span_ref"]})
    return curve


def best_candidate(curve: list[dict]) -> dict:
    return min(curve, key=lambda row: (-row["exact_f1"],
                                       abs(row["threshold"] - base.DEFAULT),
                                       row["step"], row["threshold"]))


def select_model(bundle: Path, run: Path, model_dir: Path,
                 tokenizer_dir: Path, val: list[dict]) -> tuple[dict, Path]:
    import torch

    trainer = run / "trainer"
    final_adapter = base.legacy.adapter_folder(trainer)
    choices = candidates(trainer, checkpoint_interval(run.name.removeprefix("source_")))
    curves = []
    adapters = {}
    for candidate in choices:
        name = candidate["candidate"]
        if candidate["checkpoint"] is None:
            adapter = final_adapter
        else:
            adapter = run / "checkpoint_adapters" / name
            checkpoints.export_checkpoint_adapter(candidate["checkpoint"], adapter,
                                                  final_adapter)
        cache = run / "validation_cache" / f"{name}.sqlite3"
        cache_binding(cache, bundle, adapter)
        print(f"Validation {run.name}/{name} step {candidate['step']}", flush=True)
        with closing(base._probability_connection(cache, create=True)) as db:
            if db.execute("SELECT COUNT(*) FROM probabilities").fetchone()[0] < len(val):
                sat = base.load_pinned_sat(model_dir, tokenizer_dir, adapter)
                infer_fixed_batches(sat, db, val, f"Validation {run.name}/{name}")
                del sat
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if db.execute("SELECT COUNT(*) FROM probabilities").fetchone()[0] != len(val):
                raise ValueError(f"Incomplete validation cache: {cache}")
            curves.extend(validation_curve(val, db, name, candidate["step"], candidate["epoch"]))
        adapters[name] = adapter
    base.write_csv(run / "checkpoint_validation.csv", curves, tuple(curves[0]))
    chosen = best_candidate(curves)
    with closing(base._probability_connection(bundle / "base_probabilities.sqlite3")) as db:
        m1_curve = validation_curve(val, db, "base", -1, -1.0)
    base.write_csv(run / "base_validation.csv", m1_curve, tuple(m1_curve[0]))
    m1 = best_candidate(m1_curve)
    selection = {"source": run.name.removeprefix("source_"), "reference": REFERENCE,
                 "criterion": "micro exact-span F1 on source validation only",
                 "n_validation": len(val),
                 "M1_threshold": m1["threshold"], "M1_validation_f1": m1["exact_f1"],
                 "M2_candidate": chosen["candidate"], "M2_step": chosen["step"],
                 "M2_epoch": chosen["epoch"], "M2_threshold": chosen["threshold"],
                 "M2_validation_f1": chosen["exact_f1"],
                 "checkpoint_validation_sha256": base.digest(run / "checkpoint_validation.csv"),
                 "base_validation_sha256": base.digest(run / "base_validation.csv"),
                 "bundle_manifest_sha256": base.digest(bundle / "manifest.json")}
    path = run / "selection.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != selection:
            raise ValueError("Existing checkpoint/threshold selection differs")
    else:
        atomic_json(path, selection)
    return selection, adapters[chosen["candidate"]]


def smoke(args: argparse.Namespace) -> None:
    """Exercise the new bundle, trainer, checkpoint export and adapted inference."""
    import torch

    bundle = Path(args.bundle).resolve()
    check_bundle(bundle)
    root = Path(args.run_root).resolve()
    marker = root / "smoke_ok.json"
    expected = run_signature(bundle)
    if marker.is_file():
        if json.loads(marker.read_text(encoding="utf-8"))["signature"] != expected:
            raise ValueError("Smoke result belongs to different code or data")
        print(f"Smoke already passed: {marker}", flush=True)
        return
    repo = Path(args.wtpsplit_repo).resolve()
    base.legacy.git_revision(repo)
    base.legacy.gpu_runtime(repo)
    model_dir, tokenizer_dir = base.pinned_backbone(bundle)
    source = base.CORPORA[0]
    run = root / "smoke"
    bind_run(run, bundle)
    original = torch.load(bundle / "folds" / f"target_{source}" / "training.pth",
                          map_location="cpu", weights_only=False)
    examples = original["fr"]["sentence"][base.DATASET_NAME]
    sample_data = {"fr": {"sentence": {base.DATASET_NAME: {
        "meta": {"train_data": examples["meta"]["train_data"][:8]},
        "data": examples["data"][:4]}}}}
    sample_path = run / "training.pth"
    if not sample_path.is_file():
        torch.save(sample_data, sample_path)
    elif torch.load(sample_path, map_location="cpu", weights_only=False) != sample_data:
        raise ValueError("Smoke training sample changed")
    trainer = run / "trainer"
    config = base.trainer_config(bundle, source, trainer, model_dir, tokenizer_dir)
    config.update({"text_path": str(sample_path), "num_train_epochs": 1,
                   "max_steps": 2, "save_steps": 1, "save_total_limit": None,
                   "load_best_model_at_end": False,
                   "per_device_train_batch_size": 2, "per_device_eval_batch_size": 2,
                   "gradient_accumulation_steps": 1, "logging_steps": 1})
    final_adapter = base.legacy.adapter_folder(trainer)
    if not checkpoints.complete_adapter(final_adapter):
        latest = checkpoints.latest_resumable_checkpoint(trainer)
        if latest:
            config["resume_from_checkpoint"] = str(latest)
        checkpoints.run_upstream(repo, config, run / "config.json")
    if not checkpoints.complete_adapter(final_adapter):
        raise RuntimeError("Smoke trainer did not save an adapter")
    choices = candidates(trainer, 1)
    intermediate = next((row for row in choices if row["checkpoint"] is not None), None)
    if intermediate is None:
        raise RuntimeError("Smoke trainer did not save an intermediate checkpoint")
    exported = run / "checkpoint_adapter"
    checkpoints.export_checkpoint_adapter(intermediate["checkpoint"], exported,
                                          final_adapter)
    val = base.read_jsonl(bundle / "folds" / f"target_{source}" / "val.jsonl")[:4]
    selections = []
    for label, adapter in (("checkpoint", exported), ("final", final_adapter)):
        cache = run / f"{label}_probabilities.sqlite3"
        cache_binding(cache, bundle, adapter)
        with closing(base._probability_connection(cache, create=True)) as db:
            sat = base.load_pinned_sat(model_dir, tokenizer_dir, adapter)
            infer_fixed_batches(sat, db, val, f"Smoke {label}")
            del sat
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            curve = validation_curve(val, db, label, intermediate["step"] if
                                     label == "checkpoint" else choices[-1]["step"], 0.0)
            selections.extend(curve)
    best = best_candidate(selections)
    atomic_json(marker, {"signature": expected, "source": source,
                         "n_train": 8, "n_validation": len(val),
                         "checkpoint": intermediate["step"],
                         "selected_candidate": best["candidate"],
                         "selected_threshold": best["threshold"],
                         "adapter_sha256": base.digest(final_adapter / "pytorch_adapter.bin")})
    print(json.dumps({"smoke_ok": True, "marker": str(marker)}, indent=2), flush=True)


def fit(args: argparse.Namespace) -> None:
    import torch

    bundle = Path(args.bundle).resolve()
    manifest = check_bundle(bundle)
    smoke_marker = Path(args.run_root).resolve() / "smoke_ok.json"
    if not smoke_marker.is_file() or json.loads(smoke_marker.read_text(
            encoding="utf-8")).get("signature") != run_signature(bundle):
        raise RuntimeError("Run the in-domain smoke job before the six GPU jobs")
    if not 0 <= args.task_index < len(base.CORPORA):
        raise ValueError("Slurm task index must be in 0..5")
    source = manifest["fold_order"][args.task_index]
    run = Path(args.run_root).resolve() / f"source_{source}"
    bind_run(run, bundle)
    repo = Path(args.wtpsplit_repo).resolve()
    revision = base.legacy.git_revision(repo)
    runtime = base.legacy.gpu_runtime(repo)
    model_dir, tokenizer_dir = base.pinned_backbone(bundle)
    folder = bundle / "folds" / f"target_{source}"
    val = base.read_jsonl(folder / "val.jsonl")
    trainer = run / "trainer"
    config = base.trainer_config(bundle, source, trainer, model_dir, tokenizer_dir)
    config.update({"save_strategy": "steps", "save_steps": checkpoint_interval(source),
                   "save_total_limit": None, "load_best_model_at_end": False})
    final_adapter = base.legacy.adapter_folder(trainer)
    if not checkpoints.complete_adapter(final_adapter):
        latest = checkpoints.latest_resumable_checkpoint(trainer)
        if latest:
            config["resume_from_checkpoint"] = str(latest)
            print(f"Resuming {source} from {latest}", flush=True)
        checkpoints.run_upstream(repo, config, run / "config.json")
    if not checkpoints.complete_adapter(final_adapter):
        raise RuntimeError(f"Final adapted model missing: {final_adapter}")
    state = trainer / "trainer_state.json"
    base.legacy.export_loss_history(state, run / "loss_history.csv")
    choices = candidates(trainer, checkpoint_interval(source))
    atomic_json(run / "training_done.json", {
        "source": source, "bundle_manifest_sha256": base.digest(bundle / "manifest.json"),
        "model_revision": model_dir.name, "tokenizer_revision": tokenizer_dir.name,
        "wtpsplit_revision": revision, "runtime": runtime,
        "checkpoint_steps": [candidate["step"] for candidate in choices],
        "configuration_sha256": base.digest(run / "config.json")})
    selection, adapter = select_model(bundle, run, model_dir, tokenizer_dir, val)
    print(json.dumps({"source": source, "selection": selection},
                     ensure_ascii=False, indent=2), flush=True)

    # Gold test records are only accessed after the selection file exists.
    tests = {target: base.read_jsonl(bundle / "folds" / f"target_{target}" / "test.jsonl")
             for target in base.CORPORA}
    test_rows = [row for target in base.CORPORA for row in tests[target]]
    cache = run / "heldout_probabilities.sqlite3"
    cache_binding(cache, bundle, adapter)
    with closing(base._probability_connection(cache, create=True)) as adapted:
        if adapted.execute("SELECT COUNT(*) FROM probabilities").fetchone()[0] < len(test_rows):
            sat = base.load_pinned_sat(model_dir, tokenizer_dir, adapter)
            infer_fixed_batches(sat, adapted, test_rows, f"Tests {source}")
            del sat
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if adapted.execute("SELECT COUNT(*) FROM probabilities").fetchone()[0] != len(test_rows):
            raise ValueError("Incomplete held-out prediction cache")
        with closing(base._probability_connection(bundle / "base_probabilities.sqlite3")) as baseline:
            detailed = []
            scores = {}
            for target in base.CORPORA:
                for row in tests[target]:
                    raw = row["raw_text"]
                    base_probs = base.get_probs(baseline, row)
                    adapted_probs = base.get_probs(adapted, row)
                    for model, threshold, probabilities in (
                        ("M0", base.DEFAULT, base_probs),
                        ("M1", selection["M1_threshold"], base_probs),
                        ("M2", selection["M2_threshold"], adapted_probs),
                    ):
                        spans = base.legacy.spans_from_probabilities(raw, probabilities, threshold)
                        detailed.append({"source": source, "target": target,
                                         "document_id": row["document_id"], "model": model,
                                         "threshold": threshold,
                                         **base.score(row, spans, details=True)})
                target_rows = [item for item in detailed if item["target"] == target]
                scores[target] = {model: base.aggregate_metrics(target_rows, model, target)
                                  for model in ("M0", "M1", "M2")}
                print(f"{source} -> {target}: "
                      f"M2 exact F1={scores[target]['M2']['span_f1']:.5f}", flush=True)
    fields = ("source", "target", "document_id", "model", "threshold", *base.SCORE_FIELDS,
              *(f"role_{role}_{suffix}" for role in base.ROLES for suffix in ("tp", "ref")),
              "fusion", "fragmented", "gold_units", "pred_segments", "supported_pred_segments")
    base.write_csv(run / "document_metrics.csv", detailed, fields)
    result = {"source": source, "reference": REFERENCE,
              "bundle_manifest_sha256": base.digest(bundle / "manifest.json"),
              "selection_sha256": base.digest(run / "selection.json"),
              "adapter": str(adapter),
              "adapter_sha256": base.digest(adapter / "pytorch_adapter.bin"),
              "head_sha256": base.digest(adapter / "pytorch_model_head.bin"),
              "test_sets": scores}
    path = run / "metrics.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != result:
        raise ValueError("Existing test metrics differ from resumed evaluation")
    atomic_json(path, result)
    print(f"Completed {source}: {path}", flush=True)


def heatmap_svg(path: Path, sources: list[str], cells: dict[tuple[str, str], float]) -> None:
    from html import escape

    cell_w, cell_h = 120, 72
    left, top = 185, 105
    width, height = left + len(sources) * cell_w + 25, top + len(sources) * cell_h + 65
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#182535}'
             '.head{font-size:13px;font-weight:600}.value{font-size:17px;font-weight:700}'
             '.title{font-size:19px;font-weight:700}</style>',
             '<text x="20" y="30" class="title">F1 des segments exacts — transfert SaT + LoRA</text>',
             '<text x="20" y="55">Lignes : domaine d’entraînement · colonnes : test indépendant</text>']
    for col, target in enumerate(sources):
        parts.append(f'<text x="{left + col * cell_w + cell_w / 2}" y="{top - 12}" '
                     f'text-anchor="middle" class="head">{escape(target)}</text>')
    for row, source in enumerate(sources):
        y = top + row * cell_h
        parts.append(f'<text x="{left - 10}" y="{y + 43}" text-anchor="end" '
                     f'class="head">{escape(source)}</text>')
        for col, target in enumerate(sources):
            value = cells[source, target]
            intensity = max(0.0, min(1.0, value))
            red = round(238 - 182 * intensity)
            green = round(246 - 118 * intensity)
            blue = round(250 - 92 * intensity)
            x = left + col * cell_w
            border = '#102c49' if source == target else '#d5e0e8'
            stroke = 3 if source == target else 1
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                         f'fill="#{red:02x}{green:02x}{blue:02x}" stroke="{border}" '
                         f'stroke-width="{stroke}"/>')
            parts.append(f'<text x="{x + cell_w / 2}" y="{y + 43}" text-anchor="middle" '
                         f'class="value">{value:.3f}</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts) + '\n', encoding='utf-8')


def summarize(args: argparse.Namespace) -> None:
    bundle = Path(args.bundle).resolve()
    manifest = check_bundle(bundle)
    root = Path(args.run_root).resolve()
    long_rows = []
    diagonal = []
    cells = {}
    for source in base.CORPORA:
        run = root / f"source_{source}"
        selection_path = run / "selection.json"
        result = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        choice = json.loads(selection_path.read_text(encoding="utf-8"))
        if (result["source"] != source or result["reference"] != REFERENCE
                or result["bundle_manifest_sha256"] != base.digest(bundle / "manifest.json")
                or result["selection_sha256"] != base.digest(selection_path)
                or choice["source"] != source):
            raise ValueError(f"Invalid run provenance: {source}")
        if set(result["test_sets"]) != set(base.CORPORA):
            raise ValueError(f"Incomplete transfer row: {source}")
        for target in base.CORPORA:
            metrics = result["test_sets"][target]
            expected_n = manifest["report"][target]["split"]["test"]
            if set(metrics) != {"M0", "M1", "M2"} or any(
                metrics[model]["n_documents"] != expected_n for model in metrics):
                raise ValueError(f"Test set differs between source models: {source}/{target}")
            for model, score in metrics.items():
                long_rows.append({"source": source, "target": target,
                                  "comparison": "in_domain" if source == target else "cross_domain",
                                  "model": model, "n_test": expected_n,
                                  "source_train": manifest["report"][source]["split"]["train"],
                                  "source_train_used": manifest["report"][source]["trainer_usable"]["train"],
                                  "source_validation": manifest["report"][source]["split"]["val"],
                                  "checkpoint": choice["M2_candidate"] if model == "M2" else "base",
                                  "threshold": (base.DEFAULT if model == "M0" else
                                                choice["M1_threshold"] if model == "M1" else
                                                choice["M2_threshold"]),
                                  **{name: score[name] for name in
                                     ("span_precision", "span_recall", "span_f1",
                                      "boundary_precision", "boundary_recall", "boundary_f1",
                                      "fusion_rate_supported", "fragmentation_rate")}})
            cells[source, target] = metrics["M2"]["span_f1"]
            if source == target:
                diagonal.append({"corpus": source, "n_train": manifest["report"][source]["split"]["train"],
                                 "n_train_used": manifest["report"][source]["trainer_usable"]["train"],
                                 "n_val": manifest["report"][source]["split"]["val"],
                                 "n_test": expected_n,
                                 "checkpoint": choice["M2_candidate"],
                                 "M1_threshold": choice["M1_threshold"],
                                 "M2_threshold": choice["M2_threshold"],
                                 **{f"{model}_exact_f1": metrics[model]["span_f1"]
                                    for model in ("M0", "M1", "M2")},
                                 "gain_M2_minus_M1": metrics["M2"]["span_f1"] -
                                                     metrics["M1"]["span_f1"]})
    base.write_csv(root / "transfer_metrics_long.csv", long_rows, tuple(long_rows[0]))
    matrix = [{"source": source, **{target: cells[source, target]
                                    for target in base.CORPORA}}
              for source in base.CORPORA]
    base.write_csv(root / "transfer_exact_f1.csv", matrix, tuple(matrix[0]))
    diagonal.append({"corpus": "MACRO_6_CORPUS", "n_train": sum(row["n_train"] for row in diagonal),
                     "n_train_used": sum(row["n_train_used"] for row in diagonal),
                     "n_val": sum(row["n_val"] for row in diagonal),
                     "n_test": sum(row["n_test"] for row in diagonal),
                     "checkpoint": "by_corpus", "M1_threshold": "by_corpus",
                     "M2_threshold": "by_corpus",
                     **{column: sum(row[column] for row in diagonal) / len(base.CORPORA)
                        for column in ("M0_exact_f1", "M1_exact_f1", "M2_exact_f1",
                                       "gain_M2_minus_M1")}})
    base.write_csv(root / "indomain_summary.csv", diagonal, tuple(diagonal[0]))
    heatmap_svg(root / "transfer_exact_f1.svg", list(base.CORPORA), cells)
    atomic_json(root / "summary_manifest.json", {
        "reference": REFERENCE, "bundle_manifest_sha256": base.digest(bundle / "manifest.json"),
        "selection_rule": "checkpoint and threshold on source validation only",
        "test_rule": "fixed 15% held-out set per target, shared by all six models",
        "input_metrics_sha256": {source: base.digest(root / f"source_{source}" / "metrics.json")
                                 for source in base.CORPORA},
        "input_selection_sha256": {source: base.digest(root / f"source_{source}" / "selection.json")
                                   for source in base.CORPORA},
        "files": {name: base.digest(root / name) for name in
                  ("transfer_metrics_long.csv", "transfer_exact_f1.csv",
                   "transfer_exact_f1.svg", "indomain_summary.csv")}})
    print(json.dumps({"summary": str(root / "indomain_summary.csv"),
                      "transfer_matrix": str(root / "transfer_exact_f1.csv"),
                      "heatmap": str(root / "transfer_exact_f1.svg")},
                     ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_cmd = commands.add_parser("prepare")
    prepare_cmd.add_argument("--source-bundle", type=Path, default=SOURCE_BUNDLE)
    prepare_cmd.add_argument("--out", type=Path, default=BUNDLE)
    prepare_cmd.set_defaults(func=prepare)
    check_cmd = commands.add_parser("check-bundle")
    check_cmd.add_argument("--bundle", type=Path, default=BUNDLE)
    check_cmd.set_defaults(func=check_command)
    smoke_cmd = commands.add_parser("smoke")
    smoke_cmd.add_argument("--bundle", type=Path, required=True)
    smoke_cmd.add_argument("--run-root", type=Path, required=True)
    smoke_cmd.add_argument("--wtpsplit-repo", type=Path, required=True)
    smoke_cmd.set_defaults(func=smoke)
    fit_cmd = commands.add_parser("fit")
    fit_cmd.add_argument("--bundle", type=Path, required=True)
    fit_cmd.add_argument("--run-root", type=Path, required=True)
    fit_cmd.add_argument("--wtpsplit-repo", type=Path, required=True)
    fit_cmd.add_argument("--task-index", type=int, required=True)
    fit_cmd.set_defaults(func=fit)
    summary_cmd = commands.add_parser("summarize")
    summary_cmd.add_argument("--bundle", type=Path, required=True)
    summary_cmd.add_argument("--run-root", type=Path, required=True)
    summary_cmd.set_defaults(func=summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
