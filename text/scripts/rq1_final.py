#!/usr/bin/env python3
"""Freeze the common narrative_v3 split and audit reusable single-source runs.

All output is private.  The old experiments are inputs, never publication scores.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import sat_indomain_v3 as single
import sat_lodo_gold_v3 as sat

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "safer_sat_indomain" / "bundle" / "sat12l_sm_gold_v3_indomain_v1"
SINGLE_RUNS = ROOT / "safer_sat_indomain" / "runs" / "sat12l_sm_gold_v3_indomain_v1"
OUTPUT = ROOT / "annotation" / "rq1_final_v1" / "bundle"
NAME = "rq1_segmentation_final_v1"
EXPECTED_GOLD = "4845dbec927c5cd1161360729382ac5497c3f5c41d1562b03d5020c2f20bdca8"
EXPECTED_COUNTS = {"btp": (4864, 730), "metallurgie": (2742, 411),
                   "caou_chimie_plas": (737, 111), "bois_textille": (1694, 254),
                   "service_temporaire": (2299, 345), "nicollin": (5591, 839)}


def sha(path: Path) -> str:
    return sat.digest(path)


def json_file(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def keys(rows: list[dict]) -> set[tuple[str, str]]:
    result = {(row["corpus"], row["document_id"]) for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate (corpus, document_id)")
    return result


def source_splits(source: Path) -> tuple[dict, dict[str, dict[str, list[dict]]]]:
    manifest = single.check_bundle(source)
    if manifest["gold_snapshot_sha256"] != EXPECTED_GOLD:
        raise ValueError("Gold snapshot changed since the single-source experiment")
    groups = {}
    text_owner = {}
    for corpus in sat.CORPORA:
        folder = source / "folds" / f"target_{corpus}"
        split = {name: sat.read_jsonl(folder / f"{name}.jsonl")
                 for name in ("train", "val", "test")}
        if len(set.union(*(keys(rows) for rows in split.values()))) != sum(map(len, split.values())):
            raise ValueError(f"Overlapping partitions: {corpus}")
        total, expected_test = EXPECTED_COUNTS[corpus]
        if sum(map(len, split.values())) != total or len(split["test"]) != expected_test:
            raise ValueError(f"Changed corpus or held-out test: {corpus}")
        for rows in split.values():
            for row in rows:
                sat.validate_gold(row)
                old = text_owner.setdefault(row["raw_text"], corpus)
                if old != corpus:
                    raise ValueError(f"Identical narrative crosses domains: {old}/{corpus}")
        groups[corpus] = split
    if (sum(sum(map(len, split.values())) for split in groups.values()) != 17927 or
            sum(len(row["units"]) for split in groups.values()
                for rows in split.values() for row in rows) != 147487):
        raise ValueError("Incomplete narrative_v3 reference")
    return manifest, groups


def split_digest(groups: dict) -> str:
    records = [(corpus, name, row["document_id"])
               for corpus in sat.CORPORA for name in ("train", "val", "test")
               for row in groups[corpus][name]]
    return hashlib.sha256(json.dumps(records, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def prepare(args: argparse.Namespace) -> None:
    import torch

    source, out = args.source.resolve(), args.out.resolve()
    old, groups = source_splits(source)
    if out.is_dir():
        print(json.dumps({"bundle": str(out), "ready": check_bundle(out)["experiment"]}))
        return
    staging = out.with_name(out.name + ".preparing")
    if staging.exists():
        # Windows can refuse a directory rename while an indexer briefly holds
        # a SQLite member. A fully verified staging tree can be committed by
        # copying it; incomplete trees are never resumed silently.
        check_bundle(staging)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            staging.rename(out)
        except PermissionError:
            shutil.copytree(staging, out)
        check_bundle(out)
        print(json.dumps({"bundle": str(out), "resumed": True}))
        return
    staging.mkdir(parents=True)
    files = {}
    report = {}
    for target in sat.CORPORA:
        folder = staging / "folds" / f"target_{target}"
        folder.mkdir(parents=True)
        split = {name: [row for corpus in sat.CORPORA if corpus != target
                        for row in groups[corpus][name]] for name in ("train", "val")}
        split["test"] = groups[target]["test"]
        if len(split["test"]) != EXPECTED_COUNTS[target][1] or any(
                row["corpus"] == target for name in ("train", "val") for row in split[name]):
            raise ValueError(f"Target leakage or wrong test: {target}")
        examples = {}
        excluded = {}
        for name, rows in split.items():
            path = folder / f"{name}.jsonl"
            sat.legacy.write_jsonl(path, rows)
            files[path.relative_to(staging).as_posix()] = sha(path)
            if name != "test":
                reasons = collections.Counter()
                examples[name] = []
                for row in rows:
                    chunks, reason, _ = sat.trainer_chunks(row)
                    if chunks is None:
                        reasons[reason] += 1
                    else:
                        examples[name].append(chunks)
                excluded[name] = dict(reasons)
                if not examples[name]:
                    raise ValueError(f"No trainer examples: {target}/{name}")
        ids = {name: {c: [r["document_id"] for r in split[name] if r["corpus"] == c]
                      for c in sat.CORPORA if c != target or name == "test"}
               for name in split}
        path = folder / "split_ids.json"
        json_file(path, ids)
        files[path.relative_to(staging).as_posix()] = sha(path)
        path = folder / "training.pth"
        torch.save({"fr": {"sentence": {sat.DATASET_NAME: {
            "meta": {"train_data": examples["train"]}, "data": examples["val"]}}}}, path)
        files[path.relative_to(staging).as_posix()] = sha(path)
        report[target] = {"train": len(split["train"]), "val": len(split["val"]),
                          "test": len(split["test"]), "trainer_excluded": excluded}
    # Preserve an immutable cache copy; SQLite backup includes any WAL transactions.
    path = staging / "base_probabilities.sqlite3"
    with sqlite3.connect(source / "base_probabilities.sqlite3") as previous, \
         sqlite3.connect(path) as current:
        previous.backup(current)
    files[path.relative_to(staging).as_posix()] = sha(path)
    path = staging / "rq1_runtime.json"
    shutil.copyfile(source / "rq1_runtime.json", path)
    files[path.relative_to(staging).as_posix()] = sha(path)
    manifest = {"experiment": NAME, "reference": sat.REFERENCE, "model": sat.MODEL,
                "seed": old["seed"], "fold_order": list(sat.CORPORA),
                "gold_snapshot_sha256": EXPECTED_GOLD,
                "common_split_sha256": split_digest(groups),
                "source_bundle_manifest_sha256": sha(source / "manifest.json"),
                "template_sha256": sha(sat.legacy.TEMPLATE),
                "thresholds": list(sat.THRESHOLDS),
                "report": report, "files": files}
    json_file(staging / "manifest.json", manifest)
    json_file(staging / "complete.json", {"manifest_sha256": sha(staging / "manifest.json")})
    check_bundle(staging)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging.rename(out)
    except PermissionError:
        shutil.copytree(staging, out)
    check_bundle(out)
    print(json.dumps({"bundle": str(out), "report": report}, ensure_ascii=False, indent=2))


def check_bundle(bundle: Path) -> dict:
    bundle = bundle.resolve()
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    complete = json.loads((bundle / "complete.json").read_text(encoding="utf-8"))
    if (complete["manifest_sha256"] != sha(manifest_path)
            or manifest["experiment"] != NAME
            or manifest["gold_snapshot_sha256"] != EXPECTED_GOLD
            or manifest["thresholds"] != list(sat.THRESHOLDS)
            or manifest["template_sha256"] != sha(sat.legacy.TEMPLATE)):
        raise ValueError("Wrong or changed final RQ1 bundle")
    for relative, expected in manifest["files"].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle) or not path.is_file() or sha(path) != expected:
            raise ValueError(f"Missing or altered bundle member: {relative}")
    seen_tests = set()
    source_validation = {}
    for target in sat.CORPORA:
        folder = bundle / "folds" / f"target_{target}"
        groups = {name: sat.read_jsonl(folder / f"{name}.jsonl")
                  for name in ("train", "val", "test")}
        ids = {name: keys(rows) for name, rows in groups.items()}
        if any(not ids[name] for name in ids) or any(ids[a] & ids[b] for a, b in
               (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError(f"Empty or leaking partition: {target}")
        if any(row["corpus"] == target for name in ("train", "val") for row in groups[name]):
            raise ValueError(f"Target used for learning: {target}")
        if any(row["corpus"] != target for row in groups["test"]):
            raise ValueError(f"Wrong target test: {target}")
        expected_ids = json.loads((folder / "split_ids.json").read_text(encoding="utf-8"))
        for name, rows in groups.items():
            for corpus in sat.CORPORA:
                observed = [row["document_id"] for row in rows if row["corpus"] == corpus]
                if observed != expected_ids[name].get(corpus, []):
                    raise ValueError(f"Split identifiers changed: {target}/{name}/{corpus}")
            for row in rows:
                sat.validate_gold(row)
        for source in sat.CORPORA:
            if source == target:
                continue
            observed = {row["document_id"] for row in groups["val"] if row["corpus"] == source}
            if source in source_validation and observed != source_validation[source]:
                raise ValueError(f"Source validation differs by fold: {source}")
            source_validation[source] = observed
        if len(groups["test"]) != EXPECTED_COUNTS[target][1]:
            raise ValueError(f"Wrong fixed test size: {target}")
        seen_tests.update(ids["test"])
    if len(seen_tests) != 2690:
        raise ValueError("Fixed tests do not cover 2,690 narratives")
    return manifest


def audit_single(args: argparse.Namespace) -> None:
    bundle = args.source.resolve()
    run_root = args.runs.resolve()
    manifest, groups = source_splits(bundle)
    failures = []
    results = []
    for source in sat.CORPORA:
        run = run_root / f"source_{source}"
        try:
            selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            if selection["source"] != source or metrics["source"] != source:
                raise ValueError("Source identity mismatch")
            if (selection["bundle_manifest_sha256"] != sha(bundle / "manifest.json")
                    or metrics["bundle_manifest_sha256"] != sha(bundle / "manifest.json")
                    or metrics["selection_sha256"] != sha(run / "selection.json")):
                raise ValueError("Bundle or selection fingerprint mismatch")
            if (selection["criterion"] != "micro exact-span F1 on source validation only"
                    or selection["n_validation"] != len(groups[source]["val"])):
                raise ValueError("Selection did not use the fixed source validation")
            if (selection["checkpoint_validation_sha256"] != sha(run / "checkpoint_validation.csv")
                    or selection["base_validation_sha256"] != sha(run / "base_validation.csv")):
                raise ValueError("Source validation curve changed")
            with (run / "checkpoint_validation.csv").open(encoding="utf-8-sig", newline="") as stream:
                candidates = [{**item, "step": int(item["step"]),
                               "threshold": float(item["threshold"]),
                               "exact_f1": float(item["exact_f1"])} for item in csv.DictReader(stream)]
            best = single.best_candidate(candidates)
            if (best["candidate"] != selection["M2_candidate"] or
                    best["step"] != selection["M2_step"] or
                    best["threshold"] != selection["M2_threshold"] or
                    best["exact_f1"] != selection["M2_validation_f1"]):
                raise ValueError("LoRA checkpoint/threshold was not selected by source F1")
            with (run / "base_validation.csv").open(encoding="utf-8-sig", newline="") as stream:
                baseline = [{**item, "step": int(item["step"]),
                             "threshold": float(item["threshold"]),
                             "exact_f1": float(item["exact_f1"])} for item in csv.DictReader(stream)]
            calibrated = single.best_candidate(baseline)
            if (calibrated["threshold"] != selection["M1_threshold"] or
                    calibrated["exact_f1"] != selection["M1_validation_f1"]):
                raise ValueError("Base threshold was not selected by source F1")
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            runtime = json.loads((bundle / "rq1_runtime.json").read_text(encoding="utf-8"))
            if (Path(config["model_name_or_path"]).name != runtime["model_revision"] or
                    config["subsample"] is not None or config["report_to"] != "none"):
                raise ValueError("Unexpected mono-source training configuration")
            adapter = run / "checkpoint_adapters" / selection["M2_candidate"]
            if selection["M2_candidate"] == "final":
                adapter = run / "trainer" / "safer_lodo" / "fr"
            if (sha(adapter / "pytorch_adapter.bin") != metrics["adapter_sha256"]
                    or sha(adapter / "pytorch_model_head.bin") != metrics["head_sha256"]):
                raise ValueError("Selected adapter weights changed")
            binding = json.loads((run / "heldout_probabilities.binding.json").read_text(encoding="utf-8"))
            if (binding["adapter_sha256"] != metrics["adapter_sha256"]
                    or binding["head_sha256"] != metrics["head_sha256"]):
                raise ValueError("Held-out probabilities do not use selected weights")
            with sqlite3.connect(run / "heldout_probabilities.sqlite3") as db:
                cached = set(db.execute("SELECT corpus, document_id FROM probabilities"))
            expected = set.union(*(keys(groups[c]["test"]) for c in sat.CORPORA))
            if cached != expected:
                raise ValueError("Cached test narratives differ from fixed tests")
            if set(metrics["test_sets"]) != set(sat.CORPORA) or any(
                    metrics["test_sets"][c]["M2"]["n_documents"] != len(groups[c]["test"])
                    for c in sat.CORPORA):
                raise ValueError("Incomplete transfer scores")
            results.append({"source": source, "checkpoint": selection["M2_candidate"],
                            "threshold": selection["M2_threshold"],
                            "n_validation": selection["n_validation"],
                            "n_tests": len(cached), "adapter_sha256": metrics["adapter_sha256"]})
        except (OSError, KeyError, ValueError, sqlite3.DatabaseError) as exc:
            failures.append({"source": source, "error": str(exc)})
    report = {"reference": sat.REFERENCE, "source_bundle_sha256": sha(bundle / "manifest.json"),
              "gold_snapshot_sha256": manifest["gold_snapshot_sha256"],
              "reusable": results, "failed": failures}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        json_file(args.output, report)
    if failures:
        raise ValueError("At least one single-source model failed the audit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare")
    command.add_argument("--source", type=Path, default=SOURCE)
    command.add_argument("--out", type=Path, default=OUTPUT)
    command.set_defaults(func=prepare)
    command = commands.add_parser("check")
    command.add_argument("--bundle", type=Path, default=OUTPUT)
    command.set_defaults(func=lambda args: print(json.dumps(
        {"ready": True, "manifest": check_bundle(args.bundle)}, ensure_ascii=False, indent=2)))
    command = commands.add_parser("audit-single")
    command.add_argument("--source", type=Path, default=SOURCE)
    command.add_argument("--runs", type=Path, default=SINGLE_RUNS)
    command.add_argument("--output", type=Path)
    command.set_defaults(func=audit_single)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
