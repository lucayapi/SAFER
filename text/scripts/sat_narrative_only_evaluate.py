#!/usr/bin/env python3
"""Resume evaluation for trained narrative-only SaT+LoRA folds.

This evaluation-only recovery path preserves the original training run and
binding. Whitespace-only cleaned narratives are retained in their folds but
skipped at model inference; by construction they contain no gold units.
"""

from __future__ import annotations

import argparse
import collections
import json
from contextlib import closing
from pathlib import Path

import sat_narrative_only_lodo as m3
import sat_narrative_only_runtime as runtime


def has_model_input(row: dict) -> bool:
    return bool(row["raw_text"].strip())


def select_threshold(val: list[dict], db) -> tuple[float, list[dict]]:
    source_domains = sorted({r["corpus"] for r in val})
    if len(source_domains) != 5:
        raise ValueError("Expected five source validation domains")
    counts = collections.defaultdict(collections.Counter)
    for row in val:
        if not has_model_input(row):
            continue
        probs = runtime.get_probs(db, row)
        for threshold in runtime.THRESHOLDS:
            pred = runtime.legacy.spans_from_probabilities(row["raw_text"], probs, threshold)
            counts[row["corpus"], threshold].update(m3.score(row, pred))
    curve = []
    for threshold in runtime.THRESHOLDS:
        per_source = {source: m3.prf(counts[source, threshold])[2] for source in source_domains}
        macro = sum(per_source.values()) / len(per_source)
        curve.append({"threshold": threshold,
            "source_macro_edge_normalized_exact_f1": macro,
            **{f"f1_{source}": per_source[source] for source in source_domains}})
    best = min(curve, key=lambda row: (-row["source_macro_edge_normalized_exact_f1"],
                                      abs(row["threshold"] - runtime.DEFAULT), row["threshold"]))
    for row in curve:
        row["selected"] = row is best
    return best["threshold"], curve


def evaluate_fold(prepared: Path, run_root: Path, repo: Path, task_index: int) -> None:
    manifest = m3.check_prepared(prepared)
    if not 0 <= task_index < len(runtime.CORPORA):
        raise ValueError("task-index must be 0..5")
    target = runtime.CORPORA[task_index]
    run = run_root / f"target_{target}"
    expected_binding = m3.binding(prepared)
    binding_path = run / "binding.json"
    if not binding_path.is_file() or json.loads(binding_path.read_text(encoding="utf-8")) != expected_binding:
        raise ValueError(f"Existing M3 fold is not bound to this prepared data: {target}")
    training_done = json.loads((run / "training_done.json").read_text(encoding="utf-8"))
    if training_done.get("binding") != expected_binding:
        raise ValueError(f"Trained adapter binding mismatch: {target}")
    adapter = runtime.legacy.adapter_folder(run / "trainer")
    required_adapter_files = ("adapter_config.json", "pytorch_adapter.bin",
                              "head_config.json", "pytorch_model_head.bin")
    missing = [name for name in required_adapter_files if not (adapter / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Final LoRA adapter is incomplete for {target}: {adapter}; missing {missing}")

    fold = prepared / "folds" / f"target_{target}"
    val = runtime.read_jsonl(fold / "val.jsonl")
    test = runtime.read_jsonl(fold / "test.jsonl")
    nonblank = [row for row in val + test if has_model_input(row)]
    model_dir, tokenizer_dir = m3.pinned_backbone(prepared)
    model = runtime.load_pinned_sat(model_dir, tokenizer_dir, adapter)
    with closing(runtime._probability_connection(run / "adapted_probabilities.sqlite3", create=True)) as db:
        runtime.infer_adapted(model, db, nonblank)
        threshold, curve = select_threshold(val, db)
        runtime.verify_native(model, db, nonblank, threshold)
        runtime.write_csv(run / "source_validation_thresholds.csv", curve,
            ("threshold", "source_macro_edge_normalized_exact_f1",
             *(f"f1_{source}" for source in runtime.CORPORA if source != target), "selected"))
        totals = collections.Counter()
        with (run / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in test:
                spans = (runtime.legacy.spans_from_probabilities(row["raw_text"],
                    runtime.get_probs(db, row), threshold) if has_model_input(row) else [])
                totals.update(m3.score(row, spans))
                stream.write(json.dumps({"corpus": target, "document_id": row["document_id"],
                    "threshold": threshold,
                    "segments": [{"start": a, "end": b} for a, b in spans]},
                    ensure_ascii=False) + "\n")
    precision, recall, f1 = m3.prf(totals)
    result = {"version": m3.VERSION, "method": m3.METHOD, "target": target,
        "n_test": len(test), "seed": runtime.SEED, "threshold": threshold,
        "selection": "source validation only; five-domain macro edge-normalized exact F1",
        "normalization": "trim Unicode whitespace and punctuation at span edges only",
        "counts": dict(totals), "metrics": {"precision": precision, "recall": recall,
            "edge_normalized_exact_f1": f1}, "completed_at_utc": runtime.stamp(),
        "evaluation_recovery": "whitespace-only narrative inputs skipped; no training repeated"}
    runtime.write_json(run / "metrics.json", result)
    runtime.write_json(run / "evaluation_recovery.json", {
        "binding": expected_binding, "target": target,
        "blank_only_rows_skipped_for_inference": sum(not has_model_input(row) for row in val + test),
        "completed_at_utc": runtime.stamp()})
    print(json.dumps({"target": target, "n_test": len(test), "threshold": threshold,
        "precision": precision, "recall": recall,
        "edge_normalized_exact_f1": f1}, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=m3.PREPARED)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--wtpsplit-repo", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    args = parser.parse_args()
    runtime.legacy.gpu_runtime(args.wtpsplit_repo.resolve())
    evaluate_fold(args.prepared.resolve(), args.run_root.resolve(),
                  args.wtpsplit_repo.resolve(), args.task_index)


if __name__ == "__main__":
    main()
