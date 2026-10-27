#!/usr/bin/env python3
"""Rescore saved M2/M3 test spans at fixed IoU matching thresholds.

Model checkpoints, boundary thresholds and test predictions remain unchanged.
The output is an unweighted six-domain macro average of per-domain span F1.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

import compare_narrative_only_saved_runs as base


ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = (0.3, 0.5, 0.7)
METHODS = ("M2", "M3")


def matches_at_threshold(adjacency: list[list[int]], n_gold: int) -> int:
    owner = [-1] * n_gold

    def augment(pred_index: int, seen: set[int]) -> bool:
        for gold_index in adjacency[pred_index]:
            if gold_index in seen:
                continue
            seen.add(gold_index)
            if owner[gold_index] < 0 or augment(owner[gold_index], seen):
                owner[gold_index] = pred_index
                return True
        return False

    return sum(augment(index, set()) for index in range(len(adjacency)))


def threshold_counts(row: dict, predicted: list[tuple[int, int]]) -> dict[float, int]:
    raw = row["raw_text"]
    gold = [base.trim_edges(raw, int(unit["start"]), int(unit["end"]))
            for unit in row["units"]]
    gold = [span for span in gold if span[0] < span[1]]
    pred = [base.trim_edges(raw, start, end) for start, end in predicted]
    pred = [span for span in pred if span[0] < span[1]]
    ends = [end for _, end in gold]
    graphs = {threshold: [] for threshold in THRESHOLDS}
    for segment in pred:
        neighbors = {threshold: [] for threshold in THRESHOLDS}
        index = bisect.bisect_right(ends, segment[0])
        while index < len(gold) and gold[index][0] < segment[1]:
            unit = gold[index]
            overlap = base.interval_overlap(segment, unit)
            if overlap:
                union = (segment[1] - segment[0]) + (unit[1] - unit[0]) - overlap
                iou = overlap / union
                for threshold in THRESHOLDS:
                    if iou >= threshold:
                        neighbors[threshold].append(index)
            index += 1
        for threshold in THRESHOLDS:
            graphs[threshold].append(neighbors[threshold])
    return {threshold: matches_at_threshold(graphs[threshold], len(gold))
            for threshold in THRESHOLDS}


def read_comparison(path: Path) -> dict[tuple[str, str], dict]:
    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if base.digest(path) != manifest["output_sha256"]:
        raise ValueError("Saved comparison CSV does not match its provenance")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    result = {(row["target"], row["method"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate target/method in saved comparison")
    return result


def evaluate(target: str, method: str, prepared: Path, baselines: Path,
             adapted: Path, existing: dict) -> dict:
    reference = prepared / "folds" / f"target_{target}" / "test.jsonl"
    predictions = (adapted / f"target_{target}" / "predictions.jsonl" if method == "M3"
                   else baselines / "predictions" / target / "M2.jsonl")
    totals = collections.Counter()
    n_test = 0
    for row, prediction in zip(base.load_jsonl(reference),
                               base.load_jsonl(predictions), strict=True):
        if row["corpus"] != target or row.get("excluded_spans"):
            raise ValueError(f"Unexpected reference in {target}")
        spans = base.valid_spans(row, prediction, method)
        standard = base.partial_counts(row, spans)
        counts = threshold_counts(row, spans)
        if counts[0.5] != standard["iou_tp"]:
            raise ValueError(f"IoU 0.5 does not reproduce standard matching: {target}")
        if not (counts[0.3] >= counts[0.5] >= counts[0.7] >= standard["span_tp"]):
            raise ValueError(f"Non-monotone matching counts: {target}")
        totals.update(standard)
        for threshold, count in counts.items():
            totals[f"iou_{threshold}_tp"] += count
        n_test += 1
    if (n_test != int(existing["n_test"]) or
            totals["span_ref"] != int(existing["n_gold"]) or
            totals["span_pred"] != int(existing["n_pred"]) or
            totals["span_tp"] != int(existing["exact_tp"]) or
            totals["iou_0.5_tp"] != int(existing["iou_0_5_tp"])):
        raise ValueError(f"Rescoring differs from saved report: {target}/{method}")
    denominator = totals["span_pred"] + totals["span_ref"]
    result = {"target": target, "method": method, "n_test": n_test,
              "n_gold": totals["span_ref"], "n_pred": totals["span_pred"]}
    for threshold in THRESHOLDS:
        key = f"iou_0_{int(threshold * 10)}"
        result[f"{key}_tp"] = totals[f"iou_{threshold}_tp"]
        result[f"{key}_f1"] = (2 * totals[f"iou_{threshold}_tp"] / denominator
                              if denominator else 0.0)
    result["exact_tp"] = totals["span_tp"]
    result["exact_f1"] = 2 * totals["span_tp"] / denominator if denominator else 0.0
    if (abs(result["iou_0_5_f1"] - float(existing["iou_0_5_f1"])) > 1e-12 or
            abs(result["exact_f1"] - float(existing["exact_f1"])) > 1e-12):
        raise ValueError(f"Rescored F1 differs from saved report: {target}/{method}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path,
                        default=ROOT / "annotation/lodo_sat_transfer/narrative_only_lodo_v1")
    parser.add_argument("--baselines", type=Path,
                        default=ROOT / "narrative_only_lodo_v3_baselines")
    parser.add_argument("--adapted", type=Path,
                        default=ROOT / "narrative_only_lodo_checkpoint_v1")
    parser.add_argument("--comparison", type=Path,
                        default=ROOT / "reports/narrative_only_m0_m3_comparison_2026-10-26.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/narrative_only_iou_sensitivity_2026-10-27.csv")
    args = parser.parse_args()
    provenance = base.verify_provenance(args.prepared, args.baselines, args.adapted)
    existing = read_comparison(args.comparison)
    rows = [evaluate(target, method, args.prepared, args.baselines, args.adapted,
                     existing[target, method])
            for target in base.CORPORA for method in METHODS]
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        macro = {"target": "MACRO_6_CORPUS", "method": method,
                 "n_test": sum(row["n_test"] for row in group),
                 "n_gold": sum(row["n_gold"] for row in group),
                 "n_pred": sum(row["n_pred"] for row in group)}
        for key in ("iou_0_3", "iou_0_5", "iou_0_7", "exact"):
            macro[f"{key}_tp"] = sum(row[f"{key}_tp"] for row in group)
            macro[f"{key}_f1"] = sum(row[f"{key}_f1"] for row in group) / len(group)
        if abs(macro["iou_0_5_f1"] -
               float(existing["MACRO_6_CORPUS", method]["iou_0_5_f1"])) > 1e-12:
            raise ValueError("Macro IoU 0.5 does not reproduce saved report")
        rows.append(macro)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {**provenance, "thresholds": THRESHOLDS,
                "comparison_sha256": base.digest(args.comparison),
                "output_sha256": base.digest(args.output),
                "matching": "maximum-cardinality one-to-one on edge-normalized character spans",
                "selection": "saved source-selected models and boundary thresholds; target rescoring only",
                "macro": "unweighted arithmetic mean of six target-domain F1 scores"}
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "macro": [row for row in rows if row["target"] == "MACRO_6_CORPUS"]},
                     indent=2))


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
