#!/usr/bin/env python3
r"""LoRA adaptation of sat-12l-sm on the confirmed narrative_v3 RQ1 gold.

Local preparation (from SAFER):
    .\text\.venv\Scripts\python.exe text\scripts\sat_lodo_gold_v3.py prepare
    .\text\.venv\Scripts\python.exe text\scripts\sat_lodo_gold_v3.py check-bundle

The private bundle is transferred separately. Cluster training is launched by
text/jobs/sat_lodo_gold_v3_smoke.sh and text/jobs/sat_lodo_gold_v3_gpu.sh.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import random
import sqlite3
import sys
import zlib
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import sat_lodo as legacy

ROOT = Path(__file__).resolve().parents[2]
RQ1 = ROOT / "annotation" / "rq1_sat_v3_20261017"
BUNDLE = ROOT / "annotation" / "lodo_sat_transfer" / "sat12l_sm_gold_v3_rq1"
CORPORA = legacy.CORPORA
MODEL = legacy.MODEL
SEED = 20261017
DEFAULT = 0.25
DATASET_NAME = legacy.DATASET_NAME
REFERENCE = "USER_CONFIRMED_NARRATIVE_V3_GOLD"
SCORE_FIELDS = ("span_tp", "span_pred", "span_ref", "boundary_tp", "boundary_pred", "boundary_ref")
ROLES = ("A0", "A1", "B", "C", "AUTRE")
# RQ1's final grid plus its upper-end extension, used identically for M1 and M2.
THRESHOLDS = (0.0, 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005,
              0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1,
              0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8, 0.85, 0.9, 0.95,
              0.99, 0.995, 0.999, 1.0)


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    return legacy.digest(path)


def read_jsonl(path: Path) -> list[dict]:
    return legacy.read_jsonl(path)


def write_json(path: Path, value: object) -> None:
    legacy.write_json(path, value)


def write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    legacy.write_csv(path, rows, columns)


def validate_gold(row: dict) -> None:
    raw = row["raw_text"]
    if (not isinstance(raw, str) or not raw or row["corpus"] not in CORPORA
            or not isinstance(row["document_id"], str)
            or row["validation_method"] not in {
                "USER_BULK_CONFIRMATION_NARRATIVE_V3_2026-10-17", "INDIVIDUAL_APP_VALIDATION"}):
        raise ValueError("Gold record has invalid identity or provenance")
    occupied = []
    for kind, spans in (("unit", row["units"]), ("excluded", row["excluded_spans"])):
        for item in spans:
            a, b = item["start"], item["end"]
            if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(raw):
                raise ValueError(f"Invalid {kind} offset: {row['corpus']}/{row['document_id']}")
            if kind == "unit" and item["role"] not in ROLES:
                raise ValueError("Unknown gold role")
            occupied.append((a, b))
    occupied.sort()
    if any(a[1] > b[0] for a, b in zip(occupied, occupied[1:])):
        raise ValueError(f"Overlapping gold spans: {row['corpus']}/{row['document_id']}")
    cursor = 0
    for unit in row["units"]:
        if unit["start"] < cursor:
            raise ValueError("Gold units out of order")
        cursor = unit["end"]
    covered = bytearray(len(raw))
    for a, b in occupied:
        covered[a:b] = b"\x01" * (b - a)
    if any(char.isalnum() and not covered[i] for i, char in enumerate(raw)):
        raise ValueError(f"Informative gold text is uncovered: {row['corpus']}/{row['document_id']}")


def trainer_chunks(row: dict) -> tuple[list[str] | None, str, int]:
    """Use literal factual units, never label excluded text as a fact.

    The official trainer joins items using its own newline boundary marker.
    A unit with an internal newline is incompatible with this format; the
    narrative remains available for validation and target evaluation.
    """
    raw = row["raw_text"]
    units = row["units"]
    if not units:
        return None, "no_factual_units", 0
    chunks = [raw[u["start"]:u["end"]] for u in units]
    if any("\r" in chunk or "\n" in chunk for chunk in chunks):
        return None, "newline_inside_unit", 0
    omitted = len(raw) - sum(len(chunk) for chunk in chunks)
    if any(not chunk for chunk in chunks):
        raise ValueError("Empty derived trainer unit")
    return chunks, "literal_facts_only", omitted


def read_partition(path: Path, rows: list[dict]) -> dict[tuple[str, str], str]:
    partition = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for item in csv.DictReader(stream):
            key = item["corpus"], item["document_id"]
            if key in partition or item["partition"] not in {"calibration", "source_holdout"}:
                raise ValueError(f"Invalid or duplicate RQ1 partition: {key}")
            partition[key] = item["partition"]
    if set(partition) != {(row["corpus"], row["document_id"]) for row in rows}:
        raise ValueError("RQ1 partition does not exactly cover the gold snapshot")
    for corpus in CORPORA:
        states = {partition[(row["corpus"], row["document_id"])]
                  for row in rows if row["corpus"] == corpus}
        if states != {"calibration", "source_holdout"}:
            raise ValueError(f"RQ1 split is incomplete for {corpus}")
    return partition


def copy_base_probabilities(source: Path, dest: Path, keys: set[tuple[str, str]]) -> None:
    with closing(sqlite3.connect(source)) as old, closing(sqlite3.connect(dest)) as new:
        new.execute("CREATE TABLE probabilities (corpus TEXT, document_id TEXT, "
                    "n_chars INTEGER, compressed BLOB, PRIMARY KEY(corpus,document_id))")
        for record in old.execute("SELECT corpus,document_id,n_chars,compressed FROM probabilities"):
            if (record[0], record[1]) not in keys:
                raise ValueError("Old probability cache contains an unexpected document")
            new.execute("INSERT INTO probabilities VALUES (?,?,?,?)", record)
        new.commit()
        if new.execute("SELECT COUNT(*) FROM probabilities").fetchone()[0] != len(keys):
            raise ValueError("Base probability cache is incomplete")


def prepare(args: argparse.Namespace) -> None:
    rq1 = Path(args.rq1).resolve()
    out = Path(args.out).resolve()
    source_manifest = json.loads((rq1 / "manifest.json").read_text(encoding="utf-8"))
    completed = json.loads((rq1 / "completed.json").read_text(encoding="utf-8"))
    if (source_manifest["signature"]["reference"] != "narrative_v3 user-confirmed gold"
            or source_manifest["signature"]["model"] != MODEL
            or source_manifest["signature"]["split_seed"] != SEED
            or source_manifest["signature"]["source_fraction"] != 0.8
            or source_manifest["signature"]["wtpsplit_version"] != "2.2.1"
            or completed["summary_sha256"] != digest(rq1 / "summary.json")
            or source_manifest["signature"]["snapshot_sha256"] != digest(rq1 / "gold_snapshot.jsonl")):
        raise ValueError("RQ1 is not a complete, compatible narrative_v3 run")
    if out.exists():
        raise FileExistsError(f"Bundle already exists: {out}")
    rows = read_jsonl(rq1 / "gold_snapshot.jsonl")
    keys = {(row["corpus"], row["document_id"]) for row in rows}
    if len(rows) != 17927 or len(keys) != len(rows):
        raise ValueError("Unexpected gold count or duplicate IDs")
    for row in rows:
        validate_gold(row)
    partition = read_partition(rq1 / "partition_80_20.csv", rows)
    staging = out.with_name(out.name + ".preparing")
    if staging.exists():
        raise FileExistsError(f"Incomplete earlier bundle preparation: {staging}")
    staging.mkdir(parents=True)
    import torch

    files = {}
    source_report = {}
    for corpus in CORPORA:
        group = [row for row in rows if row["corpus"] == corpus]
        source_report[corpus] = {"gold_documents": len(group),
                                 "gold_units": sum(len(row["units"]) for row in group),
                                 "train_partition": sum(partition[(corpus, row["document_id"])] == "calibration"
                                                        for row in group),
                                 "validation_partition": sum(partition[(corpus, row["document_id"])] == "source_holdout"
                                                             for row in group)}
    for target in CORPORA:
        folder = staging / "folds" / f"target_{target}"
        folder.mkdir(parents=True)
        groups = {
            "train": [row for row in rows if row["corpus"] != target and
                      partition[(row["corpus"], row["document_id"])] == "calibration"],
            "val": [row for row in rows if row["corpus"] != target and
                    partition[(row["corpus"], row["document_id"])] == "source_holdout"],
            "test": [row for row in rows if row["corpus"] == target],
        }
        trainer_samples = {}
        exclusions = {}
        for name in ("train", "val"):
            examples = []
            reasons = collections.Counter()
            omitted = 0
            for row in groups[name]:
                chunks, reason, n = trainer_chunks(row)
                if chunks is None:
                    reasons[reason] += 1
                else:
                    examples.append(chunks)
                    omitted += n
            if not examples:
                raise ValueError(f"No trainer data for {target}/{name}")
            trainer_samples[name] = examples
            exclusions[name] = {"trainer_documents": len(examples),
                                "skipped": dict(reasons),
                                "nonfactual_characters_omitted": omitted}
        for name, group in groups.items():
            path = folder / f"{name}.jsonl"
            legacy.write_jsonl(path, group)
            files[path.relative_to(staging).as_posix()] = digest(path)
        ids = {name: {corpus: [r["document_id"] for r in groups[name] if r["corpus"] == corpus]
                      for corpus in CORPORA if corpus != target or name == "test"}
               for name in groups}
        path = folder / "split_ids.json"
        write_json(path, ids)
        files[path.relative_to(staging).as_posix()] = digest(path)
        dataset = {"fr": {"sentence": {DATASET_NAME: {
            "meta": {"train_data": trainer_samples["train"]},
            "data": trainer_samples["val"]}}}}
        path = folder / "training.pth"
        torch.save(dataset, path)
        files[path.relative_to(staging).as_posix()] = digest(path)
        source_report[target]["fold"] = {"train_documents": len(groups["train"]),
                                          "val_documents": len(groups["val"]),
                                          "test_documents": len(groups["test"]),
                                          "trainer": exclusions}
        print(f"{target}: train={len(groups['train'])}, val={len(groups['val'])}, "
              f"test={len(groups['test'])}", flush=True)
    path = staging / "base_probabilities.sqlite3"
    copy_base_probabilities(rq1 / "cache.sqlite3", path, keys)
    files[path.relative_to(staging).as_posix()] = digest(path)
    runtime_path = staging / "rq1_runtime.json"
    runtime_path.write_bytes((rq1 / "runtime.json").read_bytes())
    files[runtime_path.relative_to(staging).as_posix()] = digest(runtime_path)
    baseline_path = staging / "rq1_baseline_metrics.csv"
    baseline_path.write_bytes((rq1 / "rq1_main_table.csv").read_bytes())
    files[baseline_path.relative_to(staging).as_posix()] = digest(baseline_path)
    manifest = {"created_at_utc": stamp(), "reference": REFERENCE,
                "model": MODEL, "seed": SEED, "fold_order": list(CORPORA),
                "rq1_manifest_sha256": digest(rq1 / "manifest.json"),
                "gold_snapshot_sha256": digest(rq1 / "gold_snapshot.jsonl"),
                "partition_sha256": digest(rq1 / "partition_80_20.csv"),
                "template_sha256": digest(legacy.TEMPLATE),
                "train_partition": "RQ1 calibration (80% of each source)",
                "shared_validation": "RQ1 source_holdout (20% of each source)",
                "selection_metric": "macro of five source micro exact-span F1s, peripheral whitespace trimmed",
                "thresholds": list(THRESHOLDS), "default_threshold": DEFAULT,
                "source_report": source_report, "files": files}
    write_json(staging / "manifest.json", manifest)
    write_json(staging / "complete.json", {"manifest_sha256": digest(staging / "manifest.json")})
    check_bundle(staging)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(out)
    print(json.dumps({"bundle": str(out), "gold_documents": len(rows),
                      "gold_units": sum(len(row["units"]) for row in rows)}, indent=2))


def check_bundle(bundle: Path) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    complete = json.loads((bundle / "complete.json").read_text(encoding="utf-8"))
    if (complete["manifest_sha256"] != digest(bundle / "manifest.json")
            or manifest["reference"] != REFERENCE or manifest["model"] != MODEL
            or manifest["template_sha256"] != digest(legacy.TEMPLATE)
            or manifest["fold_order"] != list(CORPORA)
            or manifest["thresholds"] != list(THRESHOLDS)):
        raise ValueError("Bundle signature or LoRA configuration mismatch")
    for relative, expected in manifest["files"].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle.resolve()) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Bundled file missing or changed: {relative}")
    by_corpus: dict[str, set[str]] = {}
    for target in CORPORA:
        folder = bundle / "folds" / f"target_{target}"
        groups = {name: read_jsonl(folder / f"{name}.jsonl") for name in ("train", "val", "test")}
        ids = {name: {(row["corpus"], row["document_id"]) for row in group}
               for name, group in groups.items()}
        if any(not group or len(ids[name]) != len(group) for name, group in groups.items()):
            raise ValueError(f"Empty or duplicated fold: {target}")
        if any(ids[a] & ids[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError(f"Partition overlap: {target}")
        if any(row["corpus"] == target for row in groups["train"] + groups["val"]):
            raise ValueError(f"Target leakage: {target}")
        if any(row["corpus"] != target for row in groups["test"]):
            raise ValueError(f"Invalid target test: {target}")
        recorded = json.loads((folder / "split_ids.json").read_text(encoding="utf-8"))
        for name, group in groups.items():
            for corpus in CORPORA:
                if recorded[name].get(corpus, []) != [row["document_id"] for row in group
                                                      if row["corpus"] == corpus]:
                    raise ValueError(f"Split ID mismatch: {target}/{name}/{corpus}")
        for source in CORPORA:
            if source == target:
                continue
            ids_for_source = {item["document_id"] for item in groups["val"]
                              if item["corpus"] == source}
            if source in by_corpus and by_corpus[source] != ids_for_source:
                raise ValueError(f"Shared validation differs between rotations: {source}")
            by_corpus[source] = ids_for_source
        for row in groups["test"]:
            validate_gold(row)
        if len(groups["test"]) != manifest["source_report"][target]["gold_documents"]:
            raise ValueError(f"Incomplete target test: {target}")
    if any(len(by_corpus[corpus]) != manifest["source_report"][corpus]["validation_partition"]
           for corpus in CORPORA):
        raise ValueError("Shared source validation IDs differ across folds")
    with closing(sqlite3.connect(bundle / "base_probabilities.sqlite3")) as db:
        n = db.execute("SELECT COUNT(*) FROM probabilities").fetchone()[0]
        if n != sum(manifest["source_report"][c]["gold_documents"] for c in CORPORA):
            raise ValueError("Incomplete base model probability cache")
    return manifest


def _probability_connection(path: Path, *, create: bool = False) -> sqlite3.Connection:
    db = sqlite3.connect(path, timeout=60)
    if create:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE IF NOT EXISTS probabilities (corpus TEXT, document_id TEXT, "
                   "n_chars INTEGER, compressed BLOB, PRIMARY KEY(corpus,document_id))")
        db.commit()
    return db


def get_probs(db: sqlite3.Connection, row: dict) -> np.ndarray:
    found = db.execute("SELECT n_chars,compressed FROM probabilities WHERE corpus=? AND document_id=?",
                       (row["corpus"], row["document_id"])).fetchone()
    if found is None:
        raise ValueError(f"Missing model probabilities: {row['corpus']}/{row['document_id']}")
    probs = np.frombuffer(zlib.decompress(found[1]), dtype=np.float32)
    if found[0] != len(probs) or len(probs) != len(row["raw_text"]) or not np.isfinite(probs).all():
        raise ValueError("Model probabilities have invalid length or values")
    return probs


def infer_adapted(sat, db: sqlite3.Connection, rows: list[dict], *, chunk_size: int = 16) -> None:
    from tqdm import tqdm

    present = {(corpus, ident) for corpus, ident in
               db.execute("SELECT corpus,document_id FROM probabilities")}
    todo = [row for row in rows if (row["corpus"], row["document_id"]) not in present]
    with tqdm(total=len(todo), desc="M2 probabilités", unit="récit") as bar:
        for offset in range(0, len(todo), chunk_size):
            batch = todo[offset:offset + chunk_size]
            calculated = sat.predict_proba([row["raw_text"] for row in batch], stride=64)
            records = []
            for row, probs in zip(batch, calculated, strict=True):
                data = np.asarray(probs, dtype=np.float32)
                if len(data) != len(row["raw_text"]) or not np.isfinite(data).all():
                    raise ValueError(f"Invalid adapted probabilities: {row['corpus']}/{row['document_id']}")
                records.append((row["corpus"], row["document_id"], len(data),
                                zlib.compress(data.tobytes())))
            with db:
                db.executemany("INSERT INTO probabilities VALUES (?,?,?,?)", records)
            bar.update(len(batch))


def trim_span(raw: str, start: int, end: int) -> tuple[int, int]:
    while start < end and raw[start].isspace():
        start += 1
    while end > start and raw[end - 1].isspace():
        end -= 1
    return start, end


def normalized_spans(raw: str, spans: list[tuple[int, int]]) -> set[tuple[int, int]]:
    return {trimmed for a, b in spans if (trimmed := trim_span(raw, a, b))[0] < trimmed[1]}


def nonspace_prefix(raw: str) -> list[int]:
    prefix = [0]
    for character in raw:
        prefix.append(prefix[-1] + (not character.isspace()))
    return prefix


def boundaries(prefix: list[int], spans: list[tuple[int, int]]) -> set[int]:
    total = prefix[-1]
    return {rank for a, b in spans for position in (a, b)
            if 0 < (rank := prefix[position]) < total}


def score(row: dict, predicted: list[tuple[int, int]], *, details: bool = False) -> dict[str, int]:
    raw = row["raw_text"]
    gold_raw = [(unit["start"], unit["end"]) for unit in row["units"]]
    gold = normalized_spans(raw, gold_raw)
    pred = normalized_spans(raw, predicted)
    prefix = nonspace_prefix(raw)
    reference_boundaries = boundaries(prefix, gold_raw)
    predicted_boundaries = boundaries(prefix, predicted)
    counts = collections.Counter({
        "span_tp": len(gold & pred), "span_pred": len(pred), "span_ref": len(gold),
        "boundary_tp": len(reference_boundaries & predicted_boundaries),
        "boundary_pred": len(predicted_boundaries), "boundary_ref": len(reference_boundaries),
    })
    if details:
        pred_nonspace = [(prefix[a], prefix[b]) for a, b in pred]
        for unit in row["units"]:
            role = unit["role"]
            counts[f"role_{role}_ref"] += 1
            counts[f"role_{role}_tp"] += trim_span(raw, unit["start"], unit["end"]) in pred
            a, b = prefix[unit["start"]], prefix[unit["end"]]
            counts["fragmented"] += sum(max(a, x) < min(b, y)
                                        for x, y in pred_nonspace) >= 2
        for a, b in pred:
            n_gold = sum(a <= x and y <= b for x, y in gold)
            counts["fusion"] += n_gold >= 2
            counts["supported_pred_segments"] += n_gold >= 1
        counts["gold_units"] = len(row["units"])
        counts["pred_segments"] = len(pred)
    return dict(counts)


def prf(tp: int, predicted: int, reference: int) -> tuple[float, float, float]:
    return (tp / predicted if predicted else 0.0,
            tp / reference if reference else 0.0,
            2 * tp / (predicted + reference) if predicted + reference else 0.0)


def aggregate_metrics(rows: list[dict], model: str, target: str, *, clean_only: bool = False) -> dict:
    chosen = [row for row in rows if row["model"] == model and
              (not clean_only or row["clean_reference"])]
    counts = collections.Counter()
    for row in chosen:
        counts.update({k: v for k, v in row.items() if k in SCORE_FIELDS or k.startswith("role_")
                       or k in {"fusion", "fragmented", "gold_units", "pred_segments",
                                "supported_pred_segments"}})
    sp, sr, sf = prf(counts["span_tp"], counts["span_pred"], counts["span_ref"])
    bp, br, bf = prf(counts["boundary_tp"], counts["boundary_pred"], counts["boundary_ref"])
    return {"target": target, "model": model, "subset": "clean_reference" if clean_only else "all",
            "n_documents": len(chosen),
            **{field: counts[field] for field in SCORE_FIELDS},
            "span_precision": sp, "span_recall": sr, "span_f1": sf,
            "boundary_precision": bp, "boundary_recall": br, "boundary_f1": bf,
            "fusion_count": counts["fusion"],
            "supported_pred_segments": counts["supported_pred_segments"],
            "fusion_rate_supported": counts["fusion"] / counts["supported_pred_segments"]
            if counts["supported_pred_segments"] else 0.0,
            "fragmented_count": counts["fragmented"], "gold_units": counts["gold_units"],
            "fragmentation_rate": counts["fragmented"] / counts["gold_units"]
            if counts["gold_units"] else 0.0,
            "role_exact_recall": {role: counts[f"role_{role}_tp"] / counts[f"role_{role}_ref"]
                                  if counts[f"role_{role}_ref"] else 0.0 for role in ROLES}}


def choose_thresholds(val: list[dict], base_db: sqlite3.Connection,
                      adapted_db: sqlite3.Connection) -> tuple[dict, list[dict]]:
    from tqdm import tqdm

    source_corpora = {row["corpus"] for row in val}
    if len(source_corpora) != 5:
        raise ValueError("Exactly five source corpora are required for validation")
    grouped = {model: collections.defaultdict(collections.Counter) for model in ("M1", "M2")}
    for row in tqdm(val, desc="Validation M1 et M2", unit="récit"):
        for model, db in (("M1", base_db), ("M2", adapted_db)):
            probabilities = get_probs(db, row)
            for threshold in THRESHOLDS:
                predicted = legacy.spans_from_probabilities(row["raw_text"], probabilities, threshold)
                current = score(row, predicted)
                grouped[model][row["corpus"], threshold].update(current)
    selected = {}
    curves = []
    for model in ("M1", "M2"):
        candidates = []
        for threshold in THRESHOLDS:
            per_source = {}
            for source in source_corpora:
                current = grouped[model][source, threshold]
                if not current["span_ref"]:
                    raise ValueError(f"Empty validation reference for {source}")
                per_source[source] = prf(current["span_tp"], current["span_pred"],
                                         current["span_ref"])[2]
            candidates.append((threshold, sum(per_source.values()) / len(per_source), per_source))
        best = min(candidates, key=lambda x: (-x[1], abs(x[0] - DEFAULT), x[0]))
        selected[model] = best[0]
        for threshold, macro, per_source in candidates:
            curves.append({"model": model, "threshold": threshold,
                           "source_macro_span_f1": macro, "selected": threshold == best[0],
                           **{f"source_f1_{c}": per_source.get(c) for c in CORPORA}})
    return selected, curves


def verify_native(sat, db: sqlite3.Connection, rows: list[dict], threshold: float) -> None:
    samples = [rows[0], next((row for row in rows if "\n" in row["raw_text"]), rows[-1])]
    for row in samples:
        raw = row["raw_text"]
        native = legacy.pieces_to_spans(raw, sat.split(raw, threshold=threshold))
        cached = legacy.spans_from_probabilities(raw, get_probs(db, row), threshold)
        if native != cached:
            raise AssertionError("Cached adapted probabilities disagree with SaT.split")


def evaluate_fold(bundle: Path, target: str, run: Path, adapter: Path,
                  model_dir: Path, tokenizer_dir: Path) -> None:
    import torch
    from tqdm import tqdm

    folder = bundle / "folds" / f"target_{target}"
    val = read_jsonl(folder / "val.jsonl")
    test = read_jsonl(folder / "test.jsonl")
    sat = load_pinned_sat(model_dir, tokenizer_dir, adapter)
    with closing(_probability_connection(bundle / "base_probabilities.sqlite3")) as base_db, \
         closing(_probability_connection(run / "adapted_probabilities.sqlite3", create=True)) as adapted_db:
        infer_adapted(sat, adapted_db, val + test)
        selected, curves = choose_thresholds(val, base_db, adapted_db)
        verify_native(sat, adapted_db, val + test, selected["M2"])
        del sat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        write_csv(run / "source_validation_thresholds.csv", curves,
                  ("model", "threshold", "source_macro_span_f1", "selected",
                   *(f"source_f1_{c}" for c in CORPORA)))
        selections = {"M0": DEFAULT, **selected}
        document_rows = []
        streams = {}
        for model in selections:
            path = run / "predictions" / f"{model}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            streams[model] = path.with_suffix(".jsonl.tmp").open("w", encoding="utf-8")
        try:
            for row in tqdm(test, desc=f"Test {target}", unit="récit"):
                base_probs = get_probs(base_db, row)
                adapted_probs = get_probs(adapted_db, row)
                for model, threshold in selections.items():
                    probs = adapted_probs if model == "M2" else base_probs
                    pred = legacy.spans_from_probabilities(row["raw_text"], probs, threshold)
                    counts = score(row, pred, details=True)
                    document_rows.append({"target": target, "corpus": row["corpus"],
                                          "document_id": row["document_id"], "model": model,
                                          "threshold": threshold,
                                          "clean_reference": not row["has_issues"] and
                                          not row["has_uncertain_units"], **counts})
                    streams[model].write(json.dumps({"corpus": target, "document_id": row["document_id"],
                        "threshold": threshold,
                        "segments": [{"start": a, "end": b, "text": row["raw_text"][a:b]}
                                     for a, b in pred]}, ensure_ascii=False) + "\n")
        finally:
            for model, stream in streams.items():
                stream.close()
        for model in selections:
            path = run / "predictions" / f"{model}.jsonl"
            path.with_suffix(".jsonl.tmp").replace(path)
        fields = ("target", "corpus", "document_id", "model", "threshold", "clean_reference",
                  *SCORE_FIELDS, *(f"role_{r}_{suffix}" for r in ROLES for suffix in ("tp", "ref")),
                  "fusion", "fragmented", "gold_units", "pred_segments", "supported_pred_segments")
        write_csv(run / "document_metrics.csv", document_rows, fields)
        metrics = {model: aggregate_metrics(document_rows, model, target) for model in selections}
        clean = {model: aggregate_metrics(document_rows, model, target, clean_only=True)
                 for model in selections}
        with (bundle / "rq1_baseline_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
            old = next(row for row in csv.DictReader(stream) if row["target"] == target)
        for field, old_field in (("span_f1", "M0_span_f1"), ("boundary_f1", "M0_boundary_f1")):
            if not math.isclose(metrics["M0"][field], float(old[old_field]), abs_tol=1e-12):
                raise AssertionError(f"M0 no longer reproduces RQ1 for {target}/{field}")
        result = {"target": target, "reference": REFERENCE, "model": MODEL,
                  "bundle_manifest_sha256": digest(bundle / "manifest.json"),
                  "selection": {"train": "RQ1 calibration", "shared_source_validation": "RQ1 source_holdout",
                                "selection_metric": "source macro exact-span F1",
                                "n_validation": len(val), "M1_threshold": selected["M1"],
                                "M2_threshold": selected["M2"]},
                  "test": metrics, "clean_reference_sensitivity": clean,
                  "gain_M2_minus_M1": {metric: metrics["M2"][metric] - metrics["M1"][metric]
                                        for metric in ("span_f1", "boundary_f1")},
                  "completed_at_utc": stamp()}
        write_json(run / "metrics.json", result)
        print(json.dumps({"target": target, "thresholds": selected,
                          "M0": metrics["M0"]["span_f1"],
                          "M1": metrics["M1"]["span_f1"],
                          "M2": metrics["M2"]["span_f1"]}, indent=2))


def pinned_backbone(bundle: Path) -> tuple[Path, Path]:
    """Use the exact base model and tokenizer revisions recorded by RQ1."""
    from huggingface_hub import snapshot_download

    runtime = json.loads((bundle / "rq1_runtime.json").read_text(encoding="utf-8"))
    model_revision = runtime.get("model_revision")
    tokenizer_revision = runtime.get("tokenizer_revision")
    if not model_revision or not tokenizer_revision:
        raise ValueError("RQ1 did not record immutable model and tokenizer revisions")
    model_dir = Path(snapshot_download(repo_id="segment-any-text/sat-12l-sm",
                                       revision=model_revision)).resolve()
    tokenizer_dir = Path(snapshot_download(repo_id="FacebookAI/xlm-roberta-base",
                                           revision=tokenizer_revision)).resolve()
    if model_dir.name != model_revision or tokenizer_dir.name != tokenizer_revision:
        raise ValueError("Downloaded model/tokenizer revision differs from RQ1")
    return model_dir, tokenizer_dir


def load_pinned_sat(model_dir: Path, tokenizer_dir: Path, adapter: Path | None = None):
    import torch
    from wtpsplit import SaT

    sat = SaT(str(model_dir), tokenizer_name_or_path=str(tokenizer_dir),
              lora_path=str(adapter) if adapter else None)
    if torch.cuda.is_available():
        sat.to("cuda")
    return sat


def run_binding(run: Path, bundle: Path) -> None:
    path = run / "binding.json"
    expected = {"bundle_manifest_sha256": digest(bundle / "manifest.json"),
                "template_sha256": digest(legacy.TEMPLATE),
                "script_sha256": digest(Path(__file__))}
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Run belongs to a different bundle or code version: {run}")
    elif run.exists() and any(run.iterdir()):
        raise ValueError(f"Unbound existing run directory: {run}")
    else:
        run.mkdir(parents=True, exist_ok=True)
        write_json(path, expected)


def trainer_config(bundle: Path, target: str, output_dir: Path,
                   model_dir: Path, tokenizer_dir: Path) -> dict:
    config = legacy.base_config(bundle, target, output_dir)
    config.update({"model_name_or_path": str(model_dir),
                   "base_model": str(tokenizer_dir),
                   "seed": SEED})
    if config["subsample"] is not None or config["report_to"] != "none":
        raise ValueError("LoRA template would subsample training data or enable external logging")
    return config


def smoke(args: argparse.Namespace) -> None:
    import torch

    bundle = Path(args.bundle).resolve()
    manifest = check_bundle(bundle)
    repo = Path(args.wtpsplit_repo).resolve()
    revision = legacy.git_revision(repo)
    runtime = legacy.gpu_runtime(repo)
    model_dir, tokenizer_dir = pinned_backbone(bundle)
    run_root = Path(args.run_root).resolve()
    marker = run_root / "smoke_ok.json"
    if marker.is_file():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if (saved.get("bundle_manifest_sha256") == digest(bundle / "manifest.json")
                and saved.get("script_sha256") == digest(Path(__file__))
                and saved.get("template_sha256") == digest(legacy.TEMPLATE)):
            print(f"Smoke already passed: {marker}")
            return
        raise ValueError("Smoke marker belongs to another bundle or code version")
    target = manifest["fold_order"][0]
    smoke_dir = run_root / "smoke"
    run_binding(smoke_dir, bundle)
    original = torch.load(bundle / "folds" / f"target_{target}" / "training.pth",
                          map_location="cpu", weights_only=False)
    source = original["fr"]["sentence"][DATASET_NAME]
    dataset = {"fr": {"sentence": {DATASET_NAME: {
        "meta": {"train_data": source["meta"]["train_data"][:8]},
        "data": source["data"][:4]}}}}
    pth = smoke_dir / "training.pth"
    torch.save(dataset, pth)
    config = trainer_config(bundle, target, smoke_dir / "trainer", model_dir, tokenizer_dir)
    config.update({"text_path": str(pth), "num_train_epochs": 1, "max_steps": 1,
                   "save_steps": 1, "per_device_train_batch_size": 2,
                   "per_device_eval_batch_size": 2, "gradient_accumulation_steps": 1})
    legacy.run_trainer(repo, config, smoke_dir / "config.json")
    adapter = legacy.adapter_folder(smoke_dir / "trainer")
    if not legacy.adapter_exists(adapter):
        raise RuntimeError("Smoke did not produce a loadable LoRA adapter")
    sat = load_pinned_sat(model_dir, tokenizer_dir, adapter)
    sample = read_jsonl(bundle / "folds" / f"target_{target}" / "val.jsonl")[0]["raw_text"]
    legacy.pieces_to_spans(sample, sat.split(sample, threshold=DEFAULT))
    state = smoke_dir / "trainer" / "trainer_state.json"
    if state.is_file():
        legacy.export_loss_history(state, smoke_dir / "loss_history.csv")
    write_json(marker, {"bundle_manifest_sha256": digest(bundle / "manifest.json"),
                        "script_sha256": digest(Path(__file__)),
                        "template_sha256": digest(legacy.TEMPLATE),
                        "model_revision": model_dir.name, "tokenizer_revision": tokenizer_dir.name,
                        "wtpsplit_revision": revision, "runtime": runtime,
                        "adapter": str(adapter), "completed_at_utc": stamp()})
    print(json.dumps({"smoke_ok": True, "marker": str(marker)}, indent=2))


def fit(args: argparse.Namespace) -> None:
    bundle = Path(args.bundle).resolve()
    manifest = check_bundle(bundle)
    if not 0 <= args.task_index < len(CORPORA):
        raise ValueError("task-index must be 0..5")
    target = manifest["fold_order"][args.task_index]
    run_root = Path(args.run_root).resolve()
    smoke_marker = run_root / "smoke_ok.json"
    if not smoke_marker.is_file():
        raise RuntimeError("Run the GPU smoke job successfully before the six training jobs")
    smoke_signature = json.loads(smoke_marker.read_text(encoding="utf-8"))
    if (smoke_signature.get("bundle_manifest_sha256") != digest(bundle / "manifest.json")
            or smoke_signature.get("script_sha256") != digest(Path(__file__))
            or smoke_signature.get("template_sha256") != digest(legacy.TEMPLATE)):
        raise RuntimeError("Run the GPU smoke job successfully before the six training jobs")
    repo = Path(args.wtpsplit_repo).resolve()
    revision = legacy.git_revision(repo)
    runtime = legacy.gpu_runtime(repo)
    model_dir, tokenizer_dir = pinned_backbone(bundle)
    run = run_root / f"target_{target}"
    run_binding(run, bundle)
    config = trainer_config(bundle, target, run / "trainer", model_dir, tokenizer_dir)
    adapter = legacy.adapter_folder(run / "trainer")
    if not legacy.adapter_exists(adapter):
        checkpoints = sorted((run / "trainer").glob("checkpoint-*"),
                             key=lambda path: int(path.name.rsplit("-", 1)[-1]))
        if checkpoints:
            config["resume_from_checkpoint"] = str(checkpoints[-1])
        legacy.run_trainer(repo, config, run / "config.json")
        if not legacy.adapter_exists(adapter):
            raise RuntimeError(f"LoRA adapter not saved: {adapter}")
    state = run / "trainer" / "trainer_state.json"
    if state.is_file():
        legacy.export_loss_history(state, run / "loss_history.csv")
    write_json(run / "training_done.json", {"target": target, "reference": REFERENCE,
               "seed": SEED, "adapter": str(adapter),
               "bundle_manifest_sha256": digest(bundle / "manifest.json"),
               "config_sha256": digest(run / "config.json"),
               "model_revision": model_dir.name, "tokenizer_revision": tokenizer_dir.name,
               "wtpsplit_revision": revision, "runtime": runtime,
               "completed_at_utc": stamp()})
    if (run / "metrics.json").is_file() and not args.force_evaluate:
        print(f"Already evaluated: {run / 'metrics.json'}")
        return
    evaluate_fold(bundle, target, run, adapter, model_dir, tokenizer_dir)


def _bootstrap_fold(document_rows: list[dict], target: str, reps: int = 1000) -> dict[str, list[float]]:
    by_model = {model: {row["document_id"]: row for row in document_rows if row["model"] == model}
                for model in ("M1", "M2")}
    if set(by_model["M1"]) != set(by_model["M2"]):
        raise ValueError(f"M1/M2 target IDs differ for {target}")
    ids = sorted(by_model["M1"])
    rng = np.random.default_rng(SEED + CORPORA.index(target))
    values = {metric: [] for metric in ("span_f1", "boundary_f1")}
    for _ in range(reps):
        sample = rng.integers(0, len(ids), len(ids))
        for metric, prefix in (("span_f1", "span"), ("boundary_f1", "boundary")):
            model_f1 = {}
            for model in ("M1", "M2"):
                tp = sum(int(by_model[model][ids[int(index)]][f"{prefix}_tp"]) for index in sample)
                pred = sum(int(by_model[model][ids[int(index)]][f"{prefix}_pred"]) for index in sample)
                ref = sum(int(by_model[model][ids[int(index)]][f"{prefix}_ref"]) for index in sample)
                model_f1[model] = prf(tp, pred, ref)[2]
            values[metric].append(model_f1["M2"] - model_f1["M1"])
    return values


def summarize(args: argparse.Namespace) -> None:
    bundle = Path(args.bundle).resolve()
    manifest = check_bundle(bundle)
    root = Path(args.run_root).resolve()
    fold_results = []
    table = []
    ci_rows = []
    bootstrap = {}
    for target in CORPORA:
        run = root / f"target_{target}"
        result = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        if (result["target"] != target or result["reference"] != REFERENCE
                or result["bundle_manifest_sha256"] != digest(bundle / "manifest.json")):
            raise ValueError(f"Incompatible target result: {target}")
        fold_results.append(result)
        metrics = result["test"]
        if len({metrics[m]["n_documents"] for m in ("M0", "M1", "M2")}) != 1:
            raise ValueError(f"M0/M1/M2 target counts differ: {target}")
        table.append({"target": target, "n_test": metrics["M0"]["n_documents"],
                      "M0_threshold": DEFAULT,
                      "M1_threshold": result["selection"]["M1_threshold"],
                      "M2_threshold": result["selection"]["M2_threshold"],
                      **{f"{model}_{measure}": metrics[model][measure]
                         for model in ("M0", "M1", "M2") for measure in (
                             "span_precision", "span_recall", "span_f1",
                             "boundary_precision", "boundary_recall", "boundary_f1")},
                      "gain_M2_minus_M1_span_f1": result["gain_M2_minus_M1"]["span_f1"],
                      "gain_M2_minus_M1_boundary_f1": result["gain_M2_minus_M1"]["boundary_f1"]})
        with (run / "document_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
            document_rows = list(csv.DictReader(stream))
        if len(document_rows) != 3 * metrics["M0"]["n_documents"]:
            raise ValueError(f"Document metric row count differs: {target}")
        bootstrap[target] = _bootstrap_fold(document_rows, target, args.bootstrap)
        for metric, values in bootstrap[target].items():
            ci_rows.append({"target": target, "metric": metric,
                            "ci95_low": float(np.quantile(values, 0.025)),
                            "ci95_high": float(np.quantile(values, 0.975)),
                            "replicates": len(values)})
    numeric = [key for key in table[0] if key.startswith(("M0_", "M1_", "M2_", "gain_"))
               and not key.endswith("threshold")]
    table.append({"target": "MACRO_6_CORPUS", "n_test": sum(row["n_test"] for row in table),
                  "M0_threshold": DEFAULT, "M1_threshold": "by_target", "M2_threshold": "by_target",
                  **{key: sum(row[key] for row in table) / len(CORPORA) for key in numeric}})
    for metric in ("span_f1", "boundary_f1"):
        macro_values = [sum(bootstrap[target][metric][index] for target in CORPORA) / len(CORPORA)
                        for index in range(args.bootstrap)]
        ci_rows.append({"target": "MACRO_6_CORPUS", "metric": metric,
                        "ci95_low": float(np.quantile(macro_values, 0.025)),
                        "ci95_high": float(np.quantile(macro_values, 0.975)),
                        "replicates": args.bootstrap})
    write_csv(root / "summary.csv", table, tuple(table[0]))
    write_csv(root / "gain_ci95.csv", ci_rows,
              ("target", "metric", "ci95_low", "ci95_high", "replicates"))
    write_json(root / "summary.json", {"reference": REFERENCE,
              "bundle_manifest_sha256": digest(bundle / "manifest.json"),
              "model": MODEL, "seed": SEED, "selection": "shared source validation",
              "results": table, "n_positive_M2_minus_M1": sum(
                  row["gain_M2_minus_M1_span_f1"] > 0 for row in table[:-1]),
              "completed_at_utc": stamp()})
    print(json.dumps({"summary": str(root / "summary.csv"),
                      "macro": table[-1]}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cmd = commands.add_parser("prepare", help="Build private narrative_v3 bundle from completed RQ1")
    cmd.add_argument("--rq1", default=str(RQ1))
    cmd.add_argument("--out", default=str(BUNDLE))
    cmd.set_defaults(func=prepare)
    cmd = commands.add_parser("check-bundle")
    cmd.add_argument("--bundle", default=str(BUNDLE))
    cmd.set_defaults(func=lambda args: print(json.dumps({"ready": True, "reference": REFERENCE,
                                                         "source_report": check_bundle(Path(args.bundle).resolve())["source_report"]},
                                                        ensure_ascii=False, indent=2)))
    for name, function in (("smoke", smoke), ("fit", fit)):
        cmd = commands.add_parser(name)
        cmd.add_argument("--bundle", required=True)
        cmd.add_argument("--run-root", required=True)
        cmd.add_argument("--wtpsplit-repo", required=True)
        if name == "fit":
            cmd.add_argument("--task-index", required=True, type=int)
            cmd.add_argument("--force-evaluate", action="store_true")
        cmd.set_defaults(func=function)
    cmd = commands.add_parser("summarize")
    cmd.add_argument("--bundle", required=True)
    cmd.add_argument("--run-root", required=True)
    cmd.add_argument("--bootstrap", type=int, default=1000)
    cmd.set_defaults(func=summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
