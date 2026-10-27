#!/usr/bin/env python3
"""Audit and compare saved M0-M3 predictions on narrative-only LODO folds.

No model is trained or selected here.  All four methods are rescored with the
same Unicode edge normalization and the same held-out reference narratives.
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

from narrative_only_boundary_metric import NORMALIZATION, prf, score, trim_edges

ROOT = Path(__file__).resolve().parents[2]
CORPORA = ("btp", "metallurgie", "caou_chimie_plas", "bois_textille",
           "service_temporaire", "nicollin")
METHODS = ("M0", "M1", "M2", "M3")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def verify_provenance(prepared: Path, baselines: Path, adapted: Path) -> dict:
    prep_path = prepared / "manifest.json"
    prep_hash = digest(prep_path)
    prep = read(prep_path)
    if read(prepared / "complete.json").get("manifest_sha256") != prep_hash:
        raise ValueError("Prepared LODO bundle is incomplete")
    if prep.get("fold_order") != list(CORPORA):
        raise ValueError("Different fold order")
    base_path = baselines / "manifest.json"
    base = read(base_path)
    if (read(baselines / "complete.json").get("manifest_sha256") != digest(base_path) or
            base["binding"].get("prepared_manifest_sha256") != prep_hash):
        raise ValueError("M0-M2 results are not bound to this LODO bundle")
    for relative, expected in base["files"].items():
        file = (baselines / relative).resolve()
        if not file.is_relative_to(baselines.resolve()) or digest(file) != expected:
            raise ValueError(f"M0-M2 result changed or missing: {relative}")
    summary_path = adapted / "summary.json"
    summary = read(summary_path)
    if (summary["binding"].get("prepared_manifest_sha256") != prep_hash or
            base["binding"].get("m3_binding") != summary["binding"] or
            summary.get("seed") != base["binding"].get("seed")):
        raise ValueError("M3 and M0-M2 do not share their test bundle, seed, or model binding")
    for corpus in CORPORA:
        for part in ("test.jsonl", "split_ids.json"):
            relative = f"folds/target_{corpus}/{part}"
            if digest(prepared / relative) != prep["files"].get(relative):
                raise ValueError(f"Prepared test changed: {relative}")
        run = adapted / f"target_{corpus}"
        meta = read(run / "metrics.json")
        pred_path = run / "predictions.jsonl"
        if (read(run / "binding.json") != summary["binding"] or
                meta.get("binding") != summary["binding"] or
                meta.get("target") != corpus or
                meta.get("seed") != summary.get("seed") or
                meta.get("prediction_sha256") != digest(pred_path)):
            raise ValueError(f"M3 result changed or mismatched: {corpus}")
    return {"prepared_manifest_sha256": prep_hash,
            "baseline_manifest_sha256": digest(base_path),
            "m3_summary_sha256": digest(summary_path),
            "selection_note": "Stored source-selected checkpoints/thresholds; no retuning",
            "normalization": NORMALIZATION}


def valid_spans(row: dict, prediction: dict, method: str) -> list[tuple[int, int]]:
    if (prediction.get("corpus"), prediction.get("document_id")) != \
            (row["corpus"], row["document_id"]):
        raise ValueError(f"{method} test IDs/order differ from reference")
    raw = row["raw_text"]
    result = []
    for span in prediction["segments"]:
        start, end = span.get("start"), span.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(raw):
            raise ValueError(f"Invalid {method} offset at {row['corpus']}/{row['document_id']}")
        result.append((start, end))
    if len(result) != len(set(result)):
        raise ValueError(f"Repeated {method} span at {row['corpus']}/{row['document_id']}")
    return result


def interval_overlap(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def partial_counts(row: dict, predicted: list[tuple[int, int]]) -> dict[str, int]:
    raw = row["raw_text"]
    gold = [(trim_edges(raw, int(u["start"]), int(u["end"])), u["role"])
            for u in row["units"]]
    gold = [(span, role) for span, role in gold if span[0] < span[1]]
    if any(gold[i][0][1] > gold[i + 1][0][0] for i in range(len(gold) - 1)):
        raise ValueError(f"Overlapping or unsorted gold units: {row['corpus']}/{row['document_id']}")
    if len({span for span, _ in gold}) != len(gold):
        raise ValueError(f"Duplicate normalized gold unit: {row['corpus']}/{row['document_id']}")
    pred = [trim_edges(raw, a, b) for a, b in predicted]
    pred = [span for span in pred if span[0] < span[1]]
    if len(pred) != len(set(pred)):
        raise ValueError("Duplicate span after boundary normalization")
    ends = [span[1] for span, _ in gold]
    graph = []
    heterogeneous = 0
    all_merges = 0
    for segment in pred:
        neighbors = []
        substantially_covered_roles = []
        index = bisect.bisect_right(ends, segment[0])
        while index < len(gold) and gold[index][0][0] < segment[1]:
            unit, role = gold[index]
            overlap = interval_overlap(segment, unit)
            if overlap:
                union = (segment[1] - segment[0]) + (unit[1] - unit[0]) - overlap
                if overlap / union >= 0.5:
                    neighbors.append(index)
                if overlap * 2 >= unit[1] - unit[0]:
                    substantially_covered_roles.append(role)
            index += 1
        graph.append(neighbors)
        if len(substantially_covered_roles) >= 2:
            all_merges += 1
            if len(set(substantially_covered_roles)) >= 2:
                heterogeneous += 1
    # Maximum-cardinality one-to-one matching; an overlong prediction cannot
    # claim several reference units for the IoU F1.
    match = [-1] * len(gold)
    def augment(pred_index: int, seen: set[int]) -> bool:
        for gold_index in graph[pred_index]:
            if gold_index in seen:
                continue
            seen.add(gold_index)
            if match[gold_index] < 0 or augment(match[gold_index], seen):
                match[gold_index] = pred_index
                return True
        return False
    matches = sum(augment(index, set()) for index in range(len(pred)))
    exact = score(row, predicted)
    return {**exact, "iou_tp": matches, "iou_pred": len(pred), "iou_ref": len(gold),
            "heterogeneous_merges": heterogeneous, "all_merges": all_merges}


def evaluate_target(target: str, prepared: Path, baselines: Path, adapted: Path) -> list[dict]:
    reference = list(load_jsonl(prepared / "folds" / f"target_{target}" / "test.jsonl"))
    if any(row["corpus"] != target or row.get("excluded_spans") for row in reference):
        raise ValueError(f"Expected cleaned factual-only narratives in {target}")
    result = []
    for method in METHODS:
        path = (adapted / f"target_{target}" / "predictions.jsonl" if method == "M3" else
                baselines / "predictions" / target / f"{method}.jsonl")
        totals = collections.Counter()
        n_test = 0
        for row, prediction in zip(reference, load_jsonl(path), strict=True):
            totals.update(partial_counts(row, valid_spans(row, prediction, method)))
            n_test += 1
        if n_test != len(reference):
            raise ValueError(f"Incomplete {target}/{method} predictions")
        exact_p, exact_r, exact_f1 = prf(totals)
        iou_p, iou_r, iou_f1 = prf({"span_tp": totals["iou_tp"],
            "span_pred": totals["iou_pred"], "span_ref": totals["iou_ref"]})
        result.append({"target": target, "method": method, "n_test": n_test,
            "n_gold": totals["span_ref"], "n_pred": totals["span_pred"],
            "exact_tp": totals["span_tp"], "exact_precision": exact_p,
            "exact_recall": exact_r, "exact_f1": exact_f1,
            "iou_0_5_tp": totals["iou_tp"], "iou_0_5_precision": iou_p,
            "iou_0_5_recall": iou_r, "iou_0_5_f1": iou_f1,
            "heterogeneous_merges": totals["heterogeneous_merges"],
            "heterogeneous_merge_rate": totals["heterogeneous_merges"] / totals["span_pred"]
                                        if totals["span_pred"] else 0.0,
            "all_merges": totals["all_merges"],
            "all_merge_rate": totals["all_merges"] / totals["span_pred"]
                              if totals["span_pred"] else 0.0})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=ROOT / "annotation/lodo_sat_transfer/narrative_only_lodo_v1")
    parser.add_argument("--baselines", type=Path, default=ROOT / "narrative_only_lodo_v3_baselines")
    parser.add_argument("--adapted", type=Path, default=ROOT / "narrative_only_lodo_checkpoint_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/narrative_only_m0_m3_comparison_2026-10-26.csv")
    args = parser.parse_args()
    provenance = verify_provenance(args.prepared, args.baselines, args.adapted)
    rows = [row for target in CORPORA for row in evaluate_target(target, args.prepared, args.baselines, args.adapted)]
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        rows.append({"target": "MACRO_6_CORPUS", "method": method,
            "n_test": sum(row["n_test"] for row in group),
            "n_gold": sum(row["n_gold"] for row in group),
            "n_pred": sum(row["n_pred"] for row in group),
            "exact_tp": sum(row["exact_tp"] for row in group),
            "exact_precision": sum(row["exact_precision"] for row in group) / 6,
            "exact_recall": sum(row["exact_recall"] for row in group) / 6,
            "exact_f1": sum(row["exact_f1"] for row in group) / 6,
            "iou_0_5_tp": sum(row["iou_0_5_tp"] for row in group),
            "iou_0_5_precision": sum(row["iou_0_5_precision"] for row in group) / 6,
            "iou_0_5_recall": sum(row["iou_0_5_recall"] for row in group) / 6,
            "iou_0_5_f1": sum(row["iou_0_5_f1"] for row in group) / 6,
            "heterogeneous_merges": sum(row["heterogeneous_merges"] for row in group),
            "heterogeneous_merge_rate": sum(row["heterogeneous_merge_rate"] for row in group) / 6,
            "all_merges": sum(row["all_merges"] for row in group),
            "all_merge_rate": sum(row["all_merge_rate"] for row in group) / 6})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    metadata = {**provenance, "output_sha256": digest(args.output),
        "iou_matching": "maximum-cardinality one-to-one at character IoU >= 0.5 after edge normalization",
        "heterogeneous_merge": "predicted segment covers >=50% of each of >=2 gold factual units with different roles",
        "heterogeneous_merge_denominator": "all nonempty normalized predicted segments",
        "macro": "unweighted mean of six target-domain metrics",
        "exclusions": "Removed from narrative_only_v1; not present in gold units or merge calculation"}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "macro": [r for r in rows if r["target"] == "MACRO_6_CORPUS"]}, indent=2))


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
