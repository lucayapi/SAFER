#!/usr/bin/env python3
"""Audit gold-unit recovery by role and length from saved M2/M3 test spans.

This is a descriptive decomposition of the existing IoU >= 0.5 matching.
No model, checkpoint, or segmentation threshold is selected here.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import csv
import json
import sys
from pathlib import Path

import compare_narrative_only_saved_runs as base


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("M2", "M3")
LENGTH_BANDS = (("1-25", 1, 25), ("26-50", 26, 50),
                ("51-100", 51, 100), (">100", 101, None))


def band(length: int) -> str:
    for name, lower, upper in LENGTH_BANDS:
        if length >= lower and (upper is None or length <= upper):
            return name
    raise ValueError(f"Invalid normalized gold length: {length}")


def gold_matches(row: dict, predicted: list[tuple[int, int]]) -> tuple[list[dict], list[bool]]:
    raw = row["raw_text"]
    gold = [{"span": base.trim_edges(raw, int(unit["start"]), int(unit["end"])),
             "role": unit["role"]} for unit in row["units"]]
    gold = [unit for unit in gold if unit["span"][0] < unit["span"][1]]
    pred = [base.trim_edges(raw, start, end) for start, end in predicted]
    pred = [span for span in pred if span[0] < span[1]]
    ends = [unit["span"][1] for unit in gold]
    graph = []
    for span in pred:
        neighbors = []
        index = bisect.bisect_right(ends, span[0])
        while index < len(gold) and gold[index]["span"][0] < span[1]:
            unit = gold[index]["span"]
            overlap = base.interval_overlap(span, unit)
            if overlap:
                union = span[1] - span[0] + unit[1] - unit[0] - overlap
                if overlap / union >= 0.5:
                    neighbors.append(index)
            index += 1
        graph.append(neighbors)

    owner = [-1] * len(gold)

    def augment(pred_index: int, seen: set[int]) -> bool:
        for gold_index in graph[pred_index]:
            if gold_index in seen:
                continue
            seen.add(gold_index)
            if owner[gold_index] < 0 or augment(owner[gold_index], seen):
                owner[gold_index] = pred_index
                return True
        return False

    matches = sum(augment(index, set()) for index in range(len(graph)))
    if matches != base.partial_counts(row, predicted)["iou_tp"]:
        raise ValueError(f"Matching differs from saved scoring: {row['corpus']}/{row['document_id']}")
    return gold, [index >= 0 for index in owner]


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
                        default=ROOT / "reports/narrative_only_sat_recall_diagnostics_2026-10-27.csv")
    args = parser.parse_args()
    provenance = base.verify_provenance(args.prepared, args.baselines, args.adapted)
    with args.comparison.open(newline="", encoding="utf-8-sig") as stream:
        comparison = {(row["target"], row["method"]): row for row in csv.DictReader(stream)}
    comparison_meta = json.loads(args.comparison.with_suffix(".json").read_text(encoding="utf-8"))
    if comparison_meta["output_sha256"] != base.digest(args.comparison):
        raise ValueError("Saved comparison report changed")

    counts: dict[tuple[str, str, str, str], list[int]] = collections.defaultdict(lambda: [0, 0])
    for target in base.CORPORA:
        reference = args.prepared / "folds" / f"target_{target}" / "test.jsonl"
        for method in METHODS:
            predictions = (args.adapted / f"target_{target}" / "predictions.jsonl"
                           if method == "M3" else
                           args.baselines / "predictions" / target / "M2.jsonl")
            gold_total = matched_total = 0
            for row, prediction in zip(base.load_jsonl(reference),
                                       base.load_jsonl(predictions), strict=True):
                spans = base.valid_spans(row, prediction, method)
                gold, matched = gold_matches(row, spans)
                for unit, found in zip(gold, matched, strict=True):
                    for group, category in (("role", unit["role"]),
                                            ("length", band(unit["span"][1] - unit["span"][0]))):
                        values = counts[target, method, group, category]
                        values[0] += 1
                        values[1] += int(found)
                    gold_total += 1
                    matched_total += int(found)
            expected = comparison[target, method]
            if (gold_total != int(expected["n_gold"]) or
                    matched_total != int(expected["iou_0_5_tp"])):
                raise ValueError(f"Report totals disagree: {target}/{method}")

    rows = []
    for (target, method, group, category), (n_gold, n_matched) in sorted(counts.items()):
        rows.append({"target": target, "method": method, "group": group,
                     "category": category, "n_gold": n_gold, "n_matched": n_matched,
                     "recall": n_matched / n_gold})
    for method in METHODS:
        categories = sorted({(row["group"], row["category"]) for row in rows
                             if row["method"] == method})
        for group, category in categories:
            subset = [row for row in rows if row["method"] == method and
                      row["group"] == group and row["category"] == category]
            if len(subset) != len(base.CORPORA):
                raise ValueError(f"Category absent in a target: {method}/{group}/{category}")
            rows.append({"target": "MACRO_6_CORPUS", "method": method, "group": group,
                         "category": category,
                         "n_gold": sum(row["n_gold"] for row in subset),
                         "n_matched": sum(row["n_matched"] for row in subset),
                         "recall": sum(row["recall"] for row in subset) / len(subset)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".json").write_text(json.dumps({**provenance,
        "comparison_sha256": base.digest(args.comparison),
        "output_sha256": base.digest(args.output),
        "matching": "same maximum-cardinality one-to-one IoU >= 0.5 matching as comparison",
        "length": "number of characters in edge-normalized gold interval",
        "recall": "matched gold factual units / gold factual units in stratum",
        "macro": "unweighted mean of the six within-domain recalls; counts are pooled",
        "note": "descriptive gold-unit recovery only; no role predictions used"}, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps([row for row in rows if row["target"] == "MACRO_6_CORPUS"], indent=2))


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
