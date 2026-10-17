#!/usr/bin/env python3
"""Audit SaT training inputs, adapted weights and boundary PR by threshold.

Read-only with respect to the gold bundle and model runs. Results are written
to a separate private directory, so an in-progress checkpoint campaign is not
modified.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "text" / "vendor" / "wtpsplit"))

import sat_lodo_gold_v3 as experiment

logging.getLogger("transformers").setLevel(logging.ERROR)

DEFAULT_RUN = ROOT / "safer_sat_lodo" / "runs" / "sat12l_sm_gold_v3_rq1"
DEFAULT_OUTPUT = ROOT / "annotation" / "audit" / "sat_m2_diagnostics_20261017"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def local_tokenizer(bundle: Path, specified: str | None):
    from tokenizers import AddedToken
    from transformers import AutoTokenizer

    if specified:
        path = Path(specified).resolve()
    else:
        from huggingface_hub.constants import HF_HUB_CACHE

        revision = json.loads((bundle / "rq1_runtime.json").read_text(encoding="utf-8"))[
            "tokenizer_revision"]
        path = (Path(HF_HUB_CACHE) / "models--FacebookAI--xlm-roberta-base" /
                "snapshots" / revision)
    if not path.is_dir():
        return None, str(path)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("\n")]})
    tokenizer.model_max_length = 1_000_000  # Counting only; no model forward pass.
    return tokenizer, str(path)


def training_audit(bundle: Path, output: Path, tokenizer_dir: str | None = None) -> dict:
    import torch
    from tqdm import tqdm

    tokenizer, tokenizer_path = local_tokenizer(bundle, tokenizer_dir)
    by_corpus: dict[str, list[dict]] = {corpus: [] for corpus in experiment.CORPORA}
    seen: set[tuple[str, str]] = set()
    fold_checks = []
    for target in tqdm(experiment.CORPORA, desc="Vérification training.pth", unit="rotation"):
        fold = bundle / "folds" / f"target_{target}"
        saved = torch.load(fold / "training.pth", map_location="cpu", weights_only=False)[
            "fr"]["sentence"][experiment.DATASET_NAME]
        for partition, expected in (("train", saved["meta"]["train_data"]),
                                    ("val", saved["data"])):
            rows = experiment.read_jsonl(fold / f"{partition}.jsonl")
            derived = []
            for row in rows:
                chunks, _, _ = experiment.trainer_chunks(row)
                if chunks is not None:
                    derived.append(chunks)
                key = row["corpus"], row["document_id"]
                if key not in seen:
                    by_corpus[row["corpus"]].append(row)
                    seen.add(key)
            if derived != expected:
                raise AssertionError(f"training.pth does not match literal gold spans: {target}/{partition}")
            fold_checks.append({"target": target, "partition": partition,
                                "documents": len(rows), "trainer_documents": len(derived),
                                "literal_units_identical": True})
    counts = []
    for corpus, rows in by_corpus.items():
        stats = collections.Counter()
        token_lengths = []
        unit_lengths = []
        texts = []
        for row in rows:
            raw = row["raw_text"]
            chunks, reason, omitted = experiment.trainer_chunks(row)
            stats["documents"] += 1
            stats["original_characters"] += len(raw)
            stats["gold_units"] += len(row["units"])
            stats["documents_with_input_newline"] += "\n" in raw
            unit_lengths.extend(u["end"] - u["start"] for u in row["units"])
            if chunks is None:
                stats[f"skipped_{reason}"] += 1
                continue
            stats["training_documents"] += 1
            stats["literal_fact_characters"] += sum(map(len, chunks))
            stats["original_characters_of_training_documents"] += len(raw)
            stats["omitted_original_characters"] += omitted
            stats["inserted_newline_boundaries"] += len(chunks) - 1
            for left, right in zip(row["units"], row["units"][1:]):
                gap = raw[left["end"]:right["start"]]
                stats["adjacent_fact_pairs"] += 1
                stats["pairs_with_removed_gap"] += bool(gap)
                stats["pairs_with_removed_informative_gap"] += any(
                    char.isalnum() for char in gap)
            if omitted:
                stats["documents_with_omissions"] += 1
            texts.append("\n".join(chunks))
        if tokenizer is not None:
            for start in tqdm(range(0, len(texts), 128), desc=f"Tokens {corpus}", leave=False):
                batch = tokenizer(texts[start:start + 128], add_special_tokens=True)["input_ids"]
                token_lengths.extend(len(ids) - 2 for ids in batch)
        counts.append({"corpus": corpus, **stats,
                       "fraction_original_characters_omitted":
                           stats["omitted_original_characters"] /
                           stats["original_characters_of_training_documents"]
                           if stats["original_characters_of_training_documents"] else 0,
                       "fraction_fact_pairs_with_removed_gap":
                           stats["pairs_with_removed_gap"] / stats["adjacent_fact_pairs"]
                           if stats["adjacent_fact_pairs"] else 0,
                       "fraction_fact_pairs_with_removed_informative_gap":
                           stats["pairs_with_removed_informative_gap"] /
                           stats["adjacent_fact_pairs"] if stats["adjacent_fact_pairs"] else 0,
                       "gold_units_under_10_characters": sum(n < 10 for n in unit_lengths),
                       "gold_units_under_20_characters": sum(n < 20 for n in unit_lengths),
                       "median_gold_unit_characters": float(np.median(unit_lengths)) if unit_lengths else None,
                       "median_reconstructed_subwords": float(np.median(token_lengths))
                           if token_lengths else None,
                       "documents_over_254_subwords": sum(n > 254 for n in token_lengths)
                           if token_lengths else None,
                       "fraction_documents_over_254_subwords": sum(n > 254 for n in token_lengths) /
                           len(token_lengths) if token_lengths else None})
    path = output / "training_representation.csv"
    experiment.write_csv(path, counts, tuple(dict.fromkeys(key for row in counts for key in row)))
    write_json(output / "training_representation.json", {
        "reference": experiment.REFERENCE,
        "bundle_manifest_sha256": experiment.digest(bundle / "manifest.json"),
        "source": "literal factual units from full gold narratives, joined with synthetic newlines",
        "newline_is_removed_from_model_input_at_default_label_args": True,
        "newline_remove_prob_default": 1.0,
        "newline_boundary_label_is_retained": True,
        "original_offsets_used_as_training_labels": False,
        "training_block_size_subwords": 256,
        "training_overflow_size_subwords": 16,
        "effective_collate_context_subwords": 254,
        "one_sample_per_line": False,
        "cross_document_concatenation_in_official_group_texts": True,
        "tokenizer_snapshot": tokenizer_path if tokenizer else None,
        "fold_checks": fold_checks,
        "by_corpus": counts,
    })
    return {"by_corpus": counts, "tokenizer_available": tokenizer is not None,
            "folds_verified": len(fold_checks)}


def adapter_folder(run: Path) -> Path:
    selection = run / "checkpoint_selection.json"
    if selection.is_file():
        chosen = json.loads(selection.read_text(encoding="utf-8"))["M2_candidate"]
        if chosen != "final":
            return run / "checkpoint_adapters" / chosen
    return experiment.legacy.adapter_folder(run / "trainer")


def audit_adapter_files(run: Path) -> dict:
    folder = adapter_folder(run)
    names = ("adapter_config.json", "head_config.json", "pytorch_adapter.bin",
             "pytorch_model_head.bin")
    if any(not (folder / name).is_file() for name in names):
        raise FileNotFoundError(f"Incomplete adapted weights: {folder}")
    return {"adapter_folder": str(folder),
            "files_sha256": {name: experiment.digest(folder / name) for name in names},
            "training_config_sha256": experiment.digest(run / "config.json")}


def score_curve(rows: list[dict], db: sqlite3.Connection, corpus: str,
                split: str, selected: float) -> list[dict]:
    from tqdm import tqdm

    totals = {threshold: collections.Counter() for threshold in experiment.THRESHOLDS}
    for row in tqdm(rows, desc=f"Courbe M2 {corpus}/{split}", unit="récit"):
        probs = experiment.get_probs(db, row)
        for threshold in experiment.THRESHOLDS:
            spans = experiment.legacy.spans_from_probabilities(row["raw_text"], probs, threshold)
            counts = totals[threshold]
            counts.update(experiment.score(row, spans))
            counts["segments"] += len(spans)
            counts["segments_under_10_characters"] += sum(b - a < 10 for a, b in spans)
            counts["segments_under_20_characters"] += sum(b - a < 20 for a, b in spans)
    curve = []
    for threshold, counts in totals.items():
        bp, br, bf = experiment.prf(counts["boundary_tp"], counts["boundary_pred"],
                                     counts["boundary_ref"])
        sp, sr, sf = experiment.prf(counts["span_tp"], counts["span_pred"], counts["span_ref"])
        curve.append({"corpus": corpus, "split": split, "n_documents": len(rows),
                      "threshold": threshold, "selected": threshold == selected,
                      "boundary_precision": bp, "boundary_recall": br, "boundary_f1": bf,
                      "boundary_tp": counts["boundary_tp"],
                      "boundary_pred": counts["boundary_pred"],
                      "boundary_ref": counts["boundary_ref"],
                      "exact_span_precision": sp, "exact_span_recall": sr, "exact_span_f1": sf,
                      "segments": counts["segments"],
                      "segments_per_document": counts["segments"] / len(rows) if rows else 0,
                      "segments_under_10_characters": counts["segments_under_10_characters"],
                      "segments_under_20_characters": counts["segments_under_20_characters"]})
    return curve


def probability_difference(rows: list[dict], adapted: sqlite3.Connection,
                           baseline: sqlite3.Connection) -> dict:
    from tqdm import tqdm

    stats = {"documents_compared": 0, "documents_with_changed_probabilities": 0,
             "positions_compared": 0, "positions_changed_above_1e-6": 0,
             "sum_absolute_difference": 0.0, "max_absolute_difference": 0.0}
    for row in tqdm(rows, desc="Comparaison M0/M2", unit="récit"):
        diff = np.abs(experiment.get_probs(adapted, row) - experiment.get_probs(baseline, row))
        stats["documents_compared"] += 1
        stats["documents_with_changed_probabilities"] += bool(np.any(diff > 1e-6))
        stats["positions_compared"] += len(diff)
        stats["positions_changed_above_1e-6"] += int(np.count_nonzero(diff > 1e-6))
        stats["sum_absolute_difference"] += float(np.sum(diff))
        stats["max_absolute_difference"] = max(stats["max_absolute_difference"],
                                                 float(np.max(diff)) if len(diff) else 0.0)
    stats["mean_absolute_difference"] = (stats.pop("sum_absolute_difference") /
                                         stats["positions_compared"]
                                         if stats["positions_compared"] else 0.0)
    return stats


def plot_curves(rows: list[dict], output: Path, corpus: str, selected: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    colors = {"source_validation": "#567a98", "target_test": "#b15338"}
    for split in ("source_validation", "target_test"):
        subset = [row for row in rows if row["split"] == split]
        if not subset:
            continue
        label = ("Validation sources" if split == "source_validation" else "Test cible") + \
                f" (n={subset[0]['n_documents']})"
        color = colors[split]
        axes[0].plot([row["boundary_recall"] for row in subset],
                     [row["boundary_precision"] for row in subset],
                     "o-", markersize=3, color=color, label=label)
        axes[1].plot(range(len(subset)), [row["segments"] for row in subset],
                     "o-", markersize=3, color=color, label=label)
        chosen = next((row for row in subset if row["threshold"] == selected), None)
        if chosen:
            axes[0].scatter([chosen["boundary_recall"]], [chosen["boundary_precision"]],
                            color=color, marker="*", s=150, edgecolor="black", zorder=5)
            axes[1].scatter([subset.index(chosen)], [chosen["segments"]],
                            color=color, marker="*", s=150, edgecolor="black", zorder=5)
    axes[0].set(xlabel="Rappel des frontières", ylabel="Précision des frontières",
                xlim=(0, 1), ylim=(0, 1), title="Précision–rappel de M2")
    indices = list(range(0, len(experiment.THRESHOLDS), 4))
    if len(experiment.THRESHOLDS) - 1 not in indices:
        indices.append(len(experiment.THRESHOLDS) - 1)
    axes[1].set_xticks(indices, [f"{experiment.THRESHOLDS[i]:g}" for i in indices], rotation=45)
    axes[1].set(xlabel=f"Seuil (étoile : seuil retenu {selected:g})",
                ylabel="Nombre total de segments", title="Volume de segmentation")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle(f"SaT adapté — {corpus}")
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"figure_boundary_pr_M2_{corpus}.png", dpi=180)
    fig.savefig(output / f"figure_boundary_pr_M2_{corpus}.pdf")
    plt.close(fig)


def verify_loaded_model(bundle: Path, run: Path, rows: list[dict], cache: Path,
                        threshold: float, test_start: int) -> dict:
    """Re-infer with the original batches, whose longest text sets SaT's window."""
    import wtpsplit

    samples = sorted({0, test_start,
                      next((i for i in range(test_start, len(rows))
                            if "\n" in rows[i]["raw_text"]), len(rows) - 1),
                      len(rows) - 1})
    model_dir, tokenizer_dir = experiment.pinned_backbone(bundle)
    sat = experiment.load_pinned_sat(model_dir, tokenizer_dir, adapter_folder(run))
    checks = []
    with closing(sqlite3.connect(cache)) as db:
        for start in sorted({(index // 16) * 16 for index in samples}):
            batch = rows[start:start + 16]
            predictions = list(sat.predict_proba([row["raw_text"] for row in batch], stride=64))
            for index in samples:
                if not start <= index < start + len(batch):
                    continue
                row = rows[index]
                recalculated = np.asarray(predictions[index - start], dtype=np.float32)
                cached = experiment.get_probs(db, row)
                batch_diff = float(np.max(np.abs(recalculated - cached)))
                # Original probabilities were cached on the cluster GPU; local CPU
                # inference can differ slightly in floating-point reductions.
                if batch_diff > 1e-4:
                    raise AssertionError(f"Loaded adapter does not reproduce batched M2 cache: "
                                         f"{row['corpus']}/{row['document_id']}, max diff {batch_diff:g}")
                standalone = np.asarray(sat.predict_proba(row["raw_text"], stride=64), dtype=np.float32)
                native = experiment.legacy.pieces_to_spans(
                    row["raw_text"], sat.split(row["raw_text"], threshold=threshold))
                cached_spans = experiment.legacy.spans_from_probabilities(
                    row["raw_text"], cached, threshold)
                batched_spans = experiment.legacy.spans_from_probabilities(
                    row["raw_text"], recalculated, threshold)
                checks.append({"document_id": row["document_id"],
                               "original_batch_start": start,
                               "original_batch_size": len(batch),
                               "batched_max_absolute_probability_difference": batch_diff,
                               "batched_spans_match_cached": batched_spans == cached_spans,
                               "standalone_max_absolute_probability_difference":
                                   float(np.max(np.abs(standalone - cached))),
                               "native_split_matches_cached_spans": native == cached_spans})
    return {"status": "passed", "wtpsplit_source": wtpsplit.__file__, "checks": checks}


def curve_audit(bundle: Path, run_root: Path, output: Path,
                *, fresh_model_check: bool = False) -> dict:
    results = []
    curves = []
    missing = []
    with closing(sqlite3.connect(bundle / "base_probabilities.sqlite3")) as baseline:
        for corpus in experiment.CORPORA:
            run = run_root / f"target_{corpus}"
            metrics_path = run / "metrics.json"
            cache = run / "adapted_probabilities.sqlite3"
            if not metrics_path.is_file() or not cache.is_file():
                missing.append(corpus)
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics["target"] != corpus or metrics["reference"] != experiment.REFERENCE:
                raise ValueError(f"Run does not match the gold bundle: {run}")
            selected = metrics["selection"]["M2_threshold"]
            if selected not in experiment.THRESHOLDS:
                raise ValueError(f"Selected threshold absent from the RQ1 grid: {corpus}")
            folder = bundle / "folds" / f"target_{corpus}"
            val = experiment.read_jsonl(folder / "val.jsonl")
            test = experiment.read_jsonl(folder / "test.jsonl")
            weights = audit_adapter_files(run)
            with closing(sqlite3.connect(cache)) as adapted:
                val_curve = score_curve(val, adapted, corpus, "source_validation", selected)
                test_curve = score_curve(test, adapted, corpus, "target_test", selected)
                differences = probability_difference(test, adapted, baseline)
            fresh_check = (verify_loaded_model(bundle, run, val + test, cache, selected, len(val))
                           if fresh_model_check else {"status": "not_run"})
            selected_test = next(row for row in test_curve if row["selected"])
            official = metrics["test"]["M2"]
            for key, old in (("boundary_precision", official["boundary_precision"]),
                             ("boundary_recall", official["boundary_recall"]),
                             ("boundary_f1", official["boundary_f1"]),
                             ("exact_span_f1", official["span_f1"]),
                             ("segments", official["span_pred"])):
                if not math.isclose(selected_test[key], old, rel_tol=0, abs_tol=1e-12):
                    raise AssertionError(f"Cached M2 probabilities do not reproduce metrics: {corpus}/{key}")
            if differences["positions_changed_above_1e-6"] == 0:
                raise AssertionError(f"M2 probabilities identical to unadapted SaT: {corpus}")
            combined = val_curve + test_curve
            curves.extend(combined)
            plot_curves(combined, output, corpus, selected)
            results.append({"corpus": corpus, "selected_threshold": selected,
                            "source_validation_documents": len(val), "target_test_documents": len(test),
                            "selected_boundary_precision": selected_test["boundary_precision"],
                            "selected_boundary_recall": selected_test["boundary_recall"],
                            "selected_boundary_f1": selected_test["boundary_f1"],
                            "selected_segments": selected_test["segments"],
                            "cached_metrics_reproduced": True,
                            "weights": weights, "M2_vs_M0_probabilities": differences,
                            "fresh_loaded_adapter_check": fresh_check})
    if curves:
        experiment.write_csv(output / "boundary_pr_by_threshold.csv", curves, tuple(curves[0]))
    report = {"reference": experiment.REFERENCE,
              "bundle_manifest_sha256": experiment.digest(bundle / "manifest.json"),
              "run_root": str(run_root), "available": results, "missing": missing,
              "inference_parameters": {"model": "sat-12l-sm", "stride": 64,
                                       "block_size": 512, "min_length": 1,
                                       "max_length": None, "prior_type": "uniform",
                                       "split_on_input_newlines": True,
                                       "do_paragraph_segmentation": False,
                                       "threshold_is_explicitly_set_for_M2": True},
              "method": "cached character probabilities, same threshold-to-spans and score code as RQ1",
              "weight_check_limit": None if fresh_model_check else
                  "Only cached probability differences and adapter hashes checked; "
                  "rerun with --verify-loaded-model to re-infer selected samples."}
    write_json(output / "model_and_curve_audit.json", report)
    return {"available": [row["corpus"] for row in results], "missing": missing}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=experiment.BUNDLE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--verify-loaded-model", action="store_true",
                        help="Recompute selected samples with the saved adapter in the pinned runtime")
    args = parser.parse_args()
    bundle, run_root, output = args.bundle.resolve(), args.run_root.resolve(), args.out.resolve()
    experiment.check_bundle(bundle)
    output.mkdir(parents=True, exist_ok=True)
    training = training_audit(bundle, output, args.tokenizer_dir)
    curves = curve_audit(bundle, run_root, output,
                         fresh_model_check=args.verify_loaded_model)
    print(json.dumps({"output": str(output), "training": training,
                      "curves": curves}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
