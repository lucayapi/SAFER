#!/usr/bin/env python3
"""Evaluate M0--M2 on the exact elementary-unit protocol used by M3.

The pinned SaT probabilities and frozen LODO folds come from the private
sat12l_sm_gold_v3_rq1 bundle. No target label selects a threshold.
"""
from __future__ import annotations

import argparse
import collections
import csv
import importlib.metadata
import json
import sqlite3
from pathlib import Path

import pysbd

import rq1_pysbd_baseline as sentence_baseline
import sat_elementary_lodo as elementary
import sat_lodo_gold_v3 as sat

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "annotation" / "lodo_sat_transfer" / "sat12l_sm_gold_v3_rq1"
DEFAULT_OUTPUT = ROOT / "sat12l_sm_elementary_v1" / "baselines.csv"
METHODS = ("M0", "M1", "M2")
FIELDS = ("method", "target", "n_test", "threshold", "elementary_exact_f1",
          "factual_exact_recall", "excluded_exact_recall", *elementary.COUNT_FIELDS)


def fold_rows(bundle: Path, target: str, part: str) -> list[dict]:
    path = bundle / "folds" / f"target_{target}" / f"{part}.jsonl"
    return sat.read_jsonl(path)


def validation_counts(bundle: Path, db: sqlite3.Connection) -> dict:
    """Score each source validation corpus once, not once per target fold."""
    counts = collections.defaultdict(collections.Counter)
    seen = {}
    for target in sat.CORPORA:
        for row in fold_rows(bundle, target, "val"):
            key = (row["corpus"], row["document_id"])
            if key in seen:
                if seen[key] != row["raw_text"]:
                    raise ValueError(f"Conflicting duplicated validation row: {key}")
                continue
            seen[key] = row["raw_text"]
            probs = sat.get_probs(db, row)
            for threshold in sat.THRESHOLDS:
                pred = sat.legacy.spans_from_probabilities(
                    row["raw_text"], probs, threshold)
                counts[row["corpus"], threshold].update(elementary.score(row, pred))
    if len(seen) != sum(sat.check_bundle(bundle)["source_report"][corpus]
                        ["validation_partition"] for corpus in sat.CORPORA):
        raise ValueError("Source validation coverage is incomplete")
    return counts


def select_threshold(target: str, counts: dict) -> tuple[float, float]:
    sources = [source for source in sat.CORPORA if source != target]
    candidates = []
    for threshold in sat.THRESHOLDS:
        values = [elementary.rates(counts[source, threshold])
                  ["elementary_exact_f1"] for source in sources]
        candidates.append((threshold, sum(values) / len(values)))
    threshold, score = min(candidates, key=lambda x:
                           (-x[1], abs(x[0] - sat.DEFAULT), x[0]))
    return threshold, score


def evaluate(bundle: Path, output: Path) -> None:
    manifest = sat.check_bundle(bundle)
    cache = bundle / "base_probabilities.sqlite3"
    db = sqlite3.connect(f"file:{cache.as_posix()}?mode=ro", uri=True)
    segmenter = pysbd.Segmenter(language="fr", clean=False, char_span=True)
    try:
        selected = validation_counts(bundle, db)
        table = []
        selections = {}
        for target in sat.CORPORA:
            threshold, validation_f1 = select_threshold(target, selected)
            selections[target] = {
                "M2_threshold": threshold,
                "source_macro_elementary_exact_f1": validation_f1}
            total = {method: collections.Counter() for method in METHODS}
            test = fold_rows(bundle, target, "test")
            if len(test) != manifest["source_report"][target]["gold_documents"]:
                raise ValueError(f"Incomplete target: {target}")
            for row in test:
                raw = row["raw_text"]
                probabilities = sat.get_probs(db, row)
                predictions = {
                    "M0": sentence_baseline.pysbd_spans(raw, segmenter),
                    "M1": sat.legacy.spans_from_probabilities(
                        raw, probabilities, sat.DEFAULT),
                    "M2": sat.legacy.spans_from_probabilities(
                        raw, probabilities, threshold)}
                for method, spans in predictions.items():
                    total[method].update(elementary.score(row, spans))
            for method in METHODS:
                counts = total[method]
                table.append({"method": method, "target": target,
                              "n_test": len(test),
                              "threshold": threshold if method == "M2"
                              else sat.DEFAULT if method == "M1" else "",
                              **elementary.rates(counts), **counts})
            print(f"{target}: " + ", ".join(
                f"{method}={elementary.rates(total[method])['elementary_exact_f1']:.3f}"
                for method in METHODS), flush=True)
        for method in METHODS:
            rows = [row for row in table if row["method"] == method]
            totals = collections.Counter()
            for row in rows:
                totals.update({field: row[field] for field in elementary.COUNT_FIELDS})
            macro = {field: sum(row[field] for row in rows) / len(rows)
                     for field in elementary.rates(totals)}
            table.append({"method": method, "target": "MACRO_6_CORPUS",
                          "n_test": sum(row["n_test"] for row in rows),
                          "threshold": "source_only_by_target"
                          if method == "M2" else sat.DEFAULT if method == "M1" else "",
                          **macro, **totals})
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(table)
        metadata = {"bundle_manifest_sha256": sat.digest(bundle / "manifest.json"),
                    "base_probabilities_sha256": sat.digest(cache),
                    "script_sha256": sat.digest(Path(__file__)),
                    "pysbd_version": importlib.metadata.version("pysbd"),
                    "M1_threshold": sat.DEFAULT,
                    "M2_source_validation": selections,
                    "metrics": ["elementary_exact_f1",
                                "factual_exact_recall", "excluded_exact_recall"]}
        output.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(output, flush=True)
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    evaluate(args.bundle.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
