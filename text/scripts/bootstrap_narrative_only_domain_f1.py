#!/usr/bin/env python3
"""Paired narrative bootstrap of held-out M3-minus-M2 IoU F1.

The trained models, selected thresholds, and test predictions are fixed. Whole
narratives are sampled with replacement within each target domain; paired M2
and M3 predictions are scored on exactly the same sampled narratives.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from compare_narrative_only_saved_runs import (
    CORPORA,
    ROOT,
    digest,
    load_jsonl,
    partial_counts,
    valid_spans,
    verify_provenance,
)


def f1(true_positive: np.ndarray, predicted: np.ndarray,
       reference: np.ndarray) -> np.ndarray:
    denominator = predicted + reference
    return np.divide(2 * true_positive, denominator,
                     out=np.zeros_like(true_positive, dtype=float),
                     where=denominator != 0)


def paired_bootstrap(counts: np.ndarray, repetitions: int,
                     rng: np.random.Generator) -> dict[str, float]:
    """Columns: reference, M2 predicted/TP, M3 predicted/TP per narrative."""
    if counts.ndim != 2 or counts.shape[1] != 5 or not len(counts):
        raise ValueError("Expected nonempty per-narrative count matrix with five columns")
    if repetitions < 100:
        raise ValueError("At least 100 bootstrap replicates are required")
    totals = counts.sum(axis=0)
    point_m2 = float(f1(np.array(totals[2]), np.array(totals[1]),
                        np.array(totals[0])))
    point_m3 = float(f1(np.array(totals[4]), np.array(totals[3]),
                        np.array(totals[0])))
    sample_m2 = np.empty(repetitions, dtype=float)
    sample_m3 = np.empty(repetitions, dtype=float)
    n = len(counts)
    for start in range(0, repetitions, 128):
        end = min(start + 128, repetitions)
        sampled = rng.integers(0, n, size=(end - start, n))
        sums = counts[sampled].sum(axis=1)
        sample_m2[start:end] = f1(sums[:, 2], sums[:, 1], sums[:, 0])
        sample_m3[start:end] = f1(sums[:, 4], sums[:, 3], sums[:, 0])
    difference = sample_m3 - sample_m2
    m2_low, m2_high = np.quantile(sample_m2, (0.025, 0.975))
    m3_low, m3_high = np.quantile(sample_m3, (0.025, 0.975))
    delta_low, delta_high = np.quantile(difference, (0.025, 0.975))
    return {"n_test": n, "m2_f1": point_m2, "m3_f1": point_m3,
            "delta_f1": point_m3 - point_m2,
            "m2_ci_low": float(m2_low), "m2_ci_high": float(m2_high),
            "m3_ci_low": float(m3_low), "m3_ci_high": float(m3_high),
            "delta_ci_low": float(delta_low),
            "delta_ci_high": float(delta_high)}


def narrative_counts(target: str, prepared: Path, baselines: Path,
                     adapted: Path) -> np.ndarray:
    reference_path = prepared / "folds" / f"target_{target}" / "test.jsonl"
    m2_path = baselines / "predictions" / target / "M2.jsonl"
    m3_path = adapted / f"target_{target}" / "predictions.jsonl"
    rows = []
    for row, m2, m3 in zip(load_jsonl(reference_path), load_jsonl(m2_path),
                           load_jsonl(m3_path), strict=True):
        if row["corpus"] != target or row.get("excluded_spans"):
            raise ValueError(f"Unexpected test reference: {target}/{row['document_id']}")
        a = partial_counts(row, valid_spans(row, m2, "M2"))
        b = partial_counts(row, valid_spans(row, m3, "M3"))
        if a["iou_ref"] != b["iou_ref"]:
            raise ValueError(f"Gold denominator mismatch: {target}/{row['document_id']}")
        rows.append((a["iou_ref"], a["iou_pred"], a["iou_tp"],
                     b["iou_pred"], b["iou_tp"]))
    return np.asarray(rows, dtype=np.int64)


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
                        default=ROOT / "reports/narrative_only_m2_m3_paired_bootstrap_2026-10-26.csv")
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()

    provenance = verify_provenance(args.prepared, args.baselines, args.adapted)
    comparison_meta = json.loads(args.comparison.with_suffix(".json").read_text(encoding="utf-8"))
    if (digest(args.comparison) != comparison_meta["output_sha256"] or
            any(comparison_meta[key] != provenance[key] for key in
                ("prepared_manifest_sha256", "baseline_manifest_sha256", "m3_summary_sha256"))):
        raise ValueError("Comparison CSV or its source binding has changed")
    with args.comparison.open(newline="", encoding="utf-8-sig") as stream:
        comparison = {(r["target"], r["method"]): r for r in csv.DictReader(stream)}

    rng = np.random.default_rng(args.seed)
    results = []
    for target in CORPORA:
        counts = narrative_counts(target, args.prepared, args.baselines, args.adapted)
        result = paired_bootstrap(counts, args.repetitions, rng)
        for method, key in (("M2", "m2_f1"), ("M3", "m3_f1")):
            expected = comparison[target, method]
            if (result["n_test"] != int(expected["n_test"]) or
                    int(counts[:, 0].sum()) != int(expected["n_gold"]) or
                    abs(result[key] - float(expected["iou_0_5_f1"])) > 1e-12):
                raise ValueError(f"Recomputed {target}/{method} does not match audited F1")
        results.append({"target": target, **result})
        print(f"{target}: delta={result['delta_f1']:.4f}, "
              f"95% CI [{result['delta_ci_low']:.4f}, {result['delta_ci_high']:.4f}]",
              flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    metadata = {
        "method": "within-domain paired nonparametric bootstrap of whole test narratives",
        "estimand": "fixed-model domain-level IoU>=0.5 unit F1 and M3-minus-M2 difference",
        "interval": "95% percentile interval (2.5th and 97.5th bootstrap quantiles)",
        "repetitions": args.repetitions,
        "seed": args.seed,
        "comparison_sha256": digest(args.comparison),
        "prepared_manifest_sha256": provenance["prepared_manifest_sha256"],
        "baseline_manifest_sha256": provenance["baseline_manifest_sha256"],
        "m3_summary_sha256": provenance["m3_summary_sha256"],
        "output_sha256": digest(args.output),
        "limitation": "conditional on the single trained M3 run and its source-selected checkpoint/threshold; no training-seed or new-domain uncertainty",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
