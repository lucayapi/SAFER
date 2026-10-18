#!/usr/bin/env python3
"""Diagnostic of candidate fact boundaries from the six selected RQ1 M3 models.

This reads the retained character-probability caches; it never retrains or
performs inference. Candidate span recall is an upper bound, not segmentation
recall: it only asks whether both endpoints of a gold unit are available.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "text" / "scripts"))
import rq1_final as campaign  # noqa: E402
import sat_lodo_gold_v3 as sat  # noqa: E402

THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
INPUT_FIELDS = ("documents", "gold_boundaries", "available_boundaries",
                "score_available_boundaries", "gold_spans", "available_spans",
                "score_available_spans", "candidate_boundaries", "score_candidates",
                "forced_candidates", "forced_added", "gap_gold_spans",
                "gap_available_spans")
DETAIL_FIELDS = ("partition", "target", "corpus", "document_id", "threshold",
                 *INPUT_FIELDS)
RESULT_FIELDS = ("partition", "target", "corpus", "threshold", *INPUT_FIELDS,
                 "boundary_recall", "score_only_boundary_recall",
                 "span_upper_recall", "score_only_span_upper_recall",
                 "gap_span_upper_recall", "mean_candidates_per_document",
                 "mean_score_candidates_per_document", "mean_forced_additions_per_document")
MACRO_FIELDS = ("partition", "target", "threshold", "n_corpora", "documents",
                "macro_boundary_recall", "macro_span_upper_recall",
                "macro_gap_span_upper_recall", "macro_mean_candidates_per_document",
                "macro_mean_score_candidates_per_document",
                "macro_mean_forced_additions_per_document")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def connect_readonly(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def assert_keys(db: sqlite3.Connection, rows: list[dict], path: Path) -> None:
    expected = {(row["corpus"], row["document_id"]) for row in rows}
    actual = set(db.execute("SELECT corpus,document_id FROM probabilities"))
    if len(expected) != len(rows) or actual != expected:
        missing, extra = expected - actual, actual - expected
        raise ValueError(f"Cache IDs differ from split: {path}; "
                         f"missing={len(missing)}, extra={len(extra)}")


def checked_prediction_file(path: Path, rows: list[dict], expected_sha: str,
                            threshold: float) -> dict[str, list[tuple[int, int]]]:
    if digest(path) != expected_sha:
        raise ValueError(f"Changed M3 prediction file: {path}")
    expected = {row["document_id"]: row for row in rows}
    found: dict[str, list[tuple[int, int]]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            ident = record["document_id"]
            if ident not in expected or ident in found or record["threshold"] != threshold:
                raise ValueError(f"Unexpected or repeated M3 prediction: {path}/{ident}")
            row = expected[ident]
            if record["corpus"] != row["corpus"]:
                raise ValueError(f"Wrong corpus in {path}/{ident}")
            spans = []
            cursor = 0
            for segment in record["segments"]:
                start, end = segment["start"], segment["end"]
                if (type(start) is not int or type(end) is not int or
                        not 0 <= cursor <= start < end <= len(row["raw_text"]) or
                        row["raw_text"][start:end] != segment["text"]):
                    raise ValueError(f"Invalid M3 segment: {path}/{ident}")
                spans.append((start, end))
                cursor = end
            found[ident] = spans
    if set(found) != set(expected):
        raise ValueError(f"Incomplete M3 prediction file: {path}")
    return found


def validate_rotation(bundle: Path, runs: Path, target: str,
                      bundle_sha: str) -> tuple[list[dict], list[dict], Path, Path,
                                                dict, dict[str, list[tuple[int, int]]], dict]:
    run = runs / f"target_{target}"
    fold = bundle / "folds" / f"target_{target}"
    selection_path, metrics_path = run / "selection.json", run / "metrics.json"
    selection, metrics = read_json(selection_path), read_json(metrics_path)
    expected_binding = selection["bundle"]
    if (selection["target"] != target or metrics["target"] != target or
            expected_binding["bundle_manifest_sha256"] != bundle_sha or
            metrics["bundle"] != expected_binding or metrics["selection"] != selection or
            metrics["selection_sha256"] != digest(selection_path)):
        raise ValueError(f"Rotation provenance mismatch: {target}")
    candidate = selection["M3_candidate"]
    val_cache = run / "validation_cache" / f"{candidate}.sqlite3"
    test_cache = run / "test_probabilities.sqlite3"
    val_binding = read_json(val_cache.with_suffix(".binding.json"))
    test_binding = read_json(test_cache.with_suffix(".binding.json"))
    weights = selection["M3_weights"]
    if (val_binding != {"bundle": expected_binding, "candidate": candidate,
                        "weights": weights} or
            test_binding != {"bundle": expected_binding, "weights": weights,
                             "selected_candidate": candidate}):
        raise ValueError(f"Probability cache uses different M3 weights: {target}")
    for filename, expected_sha in weights.items():
        if digest(run / "selected_adapter" / filename) != expected_sha:
            raise ValueError(f"Selected M3 adapter changed: {target}/{filename}")
    val = sat.read_jsonl(fold / "val.jsonl")
    test = sat.read_jsonl(fold / "test.jsonl")
    if (len(val) != selection["n_validation"] or len(test) != metrics["n_test"] or
            any(row["corpus"] == target for row in val) or
            any(row["corpus"] != target for row in test)):
        raise ValueError(f"Wrong source validation or target test: {target}")
    with closing(connect_readonly(val_cache)) as db:
        assert_keys(db, val, val_cache)
    with closing(connect_readonly(test_cache)) as db:
        assert_keys(db, test, test_cache)
    prediction = run / "predictions" / "M3.jsonl"
    originals = checked_prediction_file(prediction, test,
                                        metrics["prediction_sha256"]["M3"],
                                        selection["M3_threshold"])
    sources = {str(path.relative_to(runs)): digest(path) for path in
               (selection_path, metrics_path, val_cache,
                val_cache.with_suffix(".binding.json"), test_cache,
                test_cache.with_suffix(".binding.json"), prediction,
                *(run / "selected_adapter" / name for name in weights))}
    return val, test, val_cache, test_cache, selection, originals, sources


def boundaries_from_spans(raw: str, spans: list[tuple[int, int]],
                          prefix: np.ndarray) -> set[int]:
    total = int(prefix[-1])
    return {rank for a, b in spans for pos in (a, b)
            if 0 < (rank := int(prefix[pos])) < total}


def fact_context(raw: str, units: list[dict]) -> list[bool]:
    """Flag facts adjoining an informative non-factual gap, without filling it."""
    result = []
    for index, unit in enumerate(units):
        left = 0 if index == 0 else units[index - 1]["end"]
        right = len(raw) if index + 1 == len(units) else units[index + 1]["start"]
        result.append(any(c.isalnum() for c in raw[left:unit["start"]]) or
                      any(c.isalnum() for c in raw[unit["end"]:right]))
    return result


def candidates(raw: str, probs: np.ndarray, prefix: np.ndarray,
               threshold: float, forced: set[int]) -> tuple[set[int], set[int]]:
    """The SaT probability at character i proposes a cut after i.

    Non-space character ranks implement the RQ1 boundary convention: cuts
    anywhere in the same whitespace run represent the same boundary.
    """
    total = int(prefix[-1])
    score = {int(rank) for rank in prefix[np.flatnonzero(probs > threshold) + 1]
             if 0 < rank < total}
    return score, score | forced


def document_counts(row: dict, probs: np.ndarray,
                    threshold: float) -> tuple[dict[str, int], set[int], set[int]]:
    raw = row["raw_text"]
    if len(probs) != len(raw) or not np.isfinite(probs).all() or np.any(probs < 0) or np.any(probs > 1):
        raise ValueError(f"Invalid character probabilities: {row['corpus']}/{row['document_id']}")
    prefix = np.empty(len(raw) + 1, dtype=np.int32)
    prefix[0] = 0
    prefix[1:] = np.cumsum(np.fromiter((not c.isspace() for c in raw),
                                            dtype=np.int8, count=len(raw)))
    total = int(prefix[-1])
    forced = {int(prefix[i + 1]) for i, char in enumerate(raw) if char == "\n"
              and 0 < prefix[i + 1] < total}
    score, all_candidates = candidates(raw, probs, prefix, threshold, forced)
    units = row["units"]
    if any(not 0 <= u["start"] < u["end"] <= len(raw) for u in units):
        raise ValueError(f"Invalid gold offsets: {row['corpus']}/{row['document_id']}")
    normalized = [sat.trim_span(raw, u["start"], u["end"]) for u in units]
    if any(a >= b for a, b in normalized):
        raise ValueError(f"Empty trimmed gold unit: {row['corpus']}/{row['document_id']}")
    gold = [(int(prefix[a]), int(prefix[b])) for a, b in normalized]
    gold_boundaries = {pos for a, b in gold for pos in (a, b) if 0 < pos < total}
    available = all_candidates | {0, total}
    score_available = score | {0, total}
    gaps = fact_context(raw, units)
    result = {
        "documents": 1,
        "gold_boundaries": len(gold_boundaries),
        "available_boundaries": len(gold_boundaries & all_candidates),
        "score_available_boundaries": len(gold_boundaries & score),
        "gold_spans": len(gold),
        "available_spans": sum(a in available and b in available for a, b in gold),
        "score_available_spans": sum(a in score_available and b in score_available
                                     for a, b in gold),
        "candidate_boundaries": len(all_candidates),
        "score_candidates": len(score),
        "forced_candidates": len(forced),
        "forced_added": len(forced - score),
        "gap_gold_spans": sum(gaps),
        "gap_available_spans": sum(gap and a in available and b in available
                                   for gap, (a, b) in zip(gaps, gold, strict=True)),
    }
    return result, all_candidates, set(gold_boundaries)


def ratios(counts: dict[str, int]) -> dict:
    docs = counts["documents"]
    return {
        "boundary_recall": counts["available_boundaries"] / counts["gold_boundaries"]
                           if counts["gold_boundaries"] else None,
        "score_only_boundary_recall": counts["score_available_boundaries"] /
                                      counts["gold_boundaries"]
                                      if counts["gold_boundaries"] else None,
        "span_upper_recall": counts["available_spans"] / counts["gold_spans"]
                             if counts["gold_spans"] else None,
        "score_only_span_upper_recall": counts["score_available_spans"] /
                                        counts["gold_spans"]
                                        if counts["gold_spans"] else None,
        "gap_span_upper_recall": counts["gap_available_spans"] /
                                 counts["gap_gold_spans"]
                                 if counts["gap_gold_spans"] else None,
        "mean_candidates_per_document": counts["candidate_boundaries"] / docs,
        "mean_score_candidates_per_document": counts["score_candidates"] / docs,
        "mean_forced_additions_per_document": counts["forced_added"] / docs,
    }


def write_csv(path: Path, records: list[dict], fields: tuple[str, ...]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def write_plots(summary: list[dict], macro: list[dict], selections: dict,
                out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    targets = list(sat.CORPORA)
    for axis, target in zip(axes.flat, targets, strict=True):
        for partition, name, color in (("source_validation", "Validation sources", "#14618a"),
                                       ("target_test", "Test cible", "#d47427")):
            points = sorted((row for row in summary if row["target"] == target
                             and row["partition"] == partition),
                            key=lambda row: row["macro_mean_candidates_per_document"])
            axis.plot([p["macro_mean_candidates_per_document"] for p in points],
                      [p["macro_span_upper_recall"] for p in points],
                      marker="o", linewidth=1.5, markersize=3, color=color, label=name)
            selected = next(p for p in points if
                            p["threshold"] == selections[target]["threshold"])
            axis.scatter(selected["macro_mean_candidates_per_document"],
                         selected["macro_span_upper_recall"], marker="*", s=130,
                         c=color, edgecolors="black", linewidths=0.7, zorder=5,
                         label="Seuil M3 retenu" if partition == "target_test" else None)
            for p in points:
                if p["threshold"] in (0.01, 0.1, 0.5):
                    axis.annotate(f"{p['threshold']:.2f}",
                                  (p["macro_mean_candidates_per_document"],
                                   p["macro_span_upper_recall"]), fontsize=7)
        axis.set_title(target)
        axis.set_xlabel("Frontières candidates / récit")
        axis.set_ylabel("Rappel potentiel des unités")
        axis.set_ylim(0, 1.03)
        axis.grid(alpha=0.25)
    axes.flat[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("M3 : couverture potentielle des faits selon le nombre de candidates")
    for extension in ("svg", "png"):
        fig.savefig(out / f"tradeoff_by_rotation.{extension}", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    points = sorted((row for row in macro if row["partition"] == "target_test"),
                    key=lambda row: row["macro_mean_candidates_per_document"])
    axis.plot([p["macro_mean_candidates_per_document"] for p in points],
              [p["macro_span_upper_recall"] for p in points], marker="o", color="#b24d2f")
    for p in points:
        axis.annotate(f"{p['threshold']:.2f}",
                      (p["macro_mean_candidates_per_document"],
                       p["macro_span_upper_recall"]), fontsize=8)
    axis.set(xlabel="Frontières candidates / récit (macro-domaines)",
             ylabel="Rappel potentiel des unités (macro-domaines)",
             title="M3 — six tests cibles, seuils fixés à l'avance", ylim=(0, 1.03))
    axis.grid(alpha=0.25)
    for extension in ("svg", "png"):
        fig.savefig(out / f"tradeoff_macro_test.{extension}", dpi=180)
    plt.close(fig)


def write_report(summary: list[dict], macro: list[dict], selections: dict,
                 out: Path) -> None:
    lines = [
        "# Frontières candidates de SaT + LoRA (M3)",
        "",
        "Diagnostic sur les six rotations LODO déjà entraînées, sans nouvelle inférence. "
        "Le seuil final de M3 a été sélectionné auparavant sur les validations sources. "
        "Les tests cibles ne sont utilisés ici que pour décrire la courbe, jamais pour "
        "choisir un seuil.",
        "",
        "Les probabilités sont alignées sur les caractères du récit. Une frontière est "
        "candidate lorsque sa probabilité est strictement supérieure au seuil. Les espaces "
        "sont neutralisés comme dans RQ1 ; les coupures imposées par les retours à la ligne "
        "sont ajoutées et comptées séparément. Les bornes initiale et finale du récit sont "
        "disponibles gratuitement pour le rappel potentiel des unités, mais exclues du "
        "rappel des frontières internes et du nombre de candidates.",
        "",
        "**Le rappel potentiel des unités est une borne supérieure théorique** : les deux "
        "bornes d'une unité GOLD sont disponibles, sans garantie qu'un raffineur les choisira "
        "ensemble ni qu'il évitera les fusions ou les segments parasites. Les passages non "
        "factuels informatifs restent dans le récit ; `per_corpus.csv` rapporte aussi les "
        "unités adjacentes à un tel passage.",
        "",
        "## Six tests cibles : moyenne macro-domaines",
        "",
        "| Seuil | Rappel des frontières | Rappel potentiel des unités | Candidates/récit |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in macro:
        lines.append(f"| {row['threshold']:.2f} | "
                     f"{row['macro_boundary_recall']:.1%} | "
                     f"{row['macro_span_upper_recall']:.1%} | "
                     f"{row['macro_mean_candidates_per_document']:.1f} |")
    lines.extend(["", "## Repères aux seuils M3 retenus sur les sources", "",
                  "| Cible exclue de l'entraînement | Seuil M3 | "
                  "Validation sources : rappel potentiel | "
                  "Test cible : rappel potentiel | Test : candidates/récit |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for target in sat.CORPORA:
        threshold = selections[target]["threshold"]
        source = next(row for row in summary if row["partition"] == "source_validation"
                      and row["target"] == target and row["threshold"] == threshold)
        test = next(row for row in summary if row["partition"] == "target_test"
                    and row["target"] == target and row["threshold"] == threshold)
        lines.append(f"| {target} | {threshold:.2f} | "
                     f"{source['macro_span_upper_recall']:.1%} | "
                     f"{test['macro_span_upper_recall']:.1%} | "
                     f"{test['macro_mean_candidates_per_document']:.1f} |")
    lines.extend(["", "La moyenne macro accorde le même poids à chacun des six domaines. "
                  "Les validations sources ont aussi servi à sélectionner les checkpoints "
                  "M3 ; leurs courbes sont donc descriptives, pas une évaluation indépendante.",
                  "", "Figures : `tradeoff_by_rotation.svg` et `tradeoff_macro_test.svg`. "
                  "Les comptes détaillés figurent dans `per_corpus.csv`, et les résultats "
                  "par récit dans `per_document.csv`.", ""])
    temporary = out / "RESULTS.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(out / "RESULTS.md")


def run(args: argparse.Namespace) -> None:
    bundle, runs, out = args.bundle.resolve(), args.runs.resolve(), args.output.resolve()
    manifest = campaign.check_bundle(bundle)
    bundle_sha = digest(bundle / "manifest.json")
    complete = read_json(runs / "cluster_complete.json")
    if (complete["campaign"] != campaign.NAME or
            complete["bundle_manifest_sha256"] != bundle_sha or
            complete["gold_snapshot_sha256"] != manifest["gold_snapshot_sha256"] or
            complete["common_split_sha256"] != manifest["common_split_sha256"]):
        raise ValueError("Cluster campaign and local GOLD bundle do not match")
    out.mkdir(parents=True, exist_ok=True)
    detail_path = out / "per_document.csv"
    detail_tmp = detail_path.with_name(detail_path.name + ".tmp")
    by_corpus: dict[tuple[str, str, str, float], collections.Counter] = collections.defaultdict(collections.Counter)
    input_hashes = {"bundle/manifest.json": bundle_sha,
                    "runs/cluster_complete.json": digest(runs / "cluster_complete.json")}
    selections = {}
    tests_seen = set()
    n_test = 0
    with detail_tmp.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, DETAIL_FIELDS, extrasaction="raise")
        writer.writeheader()
        for target in sat.CORPORA:
            val, test, val_cache, test_cache, selection, originals, sources = \
                validate_rotation(bundle, runs, target, bundle_sha)
            selections[target] = {"candidate": selection["M3_candidate"],
                                  "threshold": selection["M3_threshold"],
                                  "weights": selection["M3_weights"]}
            input_hashes.update({f"runs/{path}": value for path, value in sources.items()})
            for partition, rows, cache in (("source_validation", val, val_cache),
                                           ("target_test", test, test_cache)):
                with closing(connect_readonly(cache)) as db:
                    for row in tqdm(rows, desc=f"M3 {target} {partition}", unit="récit"):
                        raw = row["raw_text"]
                        probabilities = sat.get_probs(db, row)
                        if partition == "target_test":
                            test_key = (row["corpus"], row["document_id"])
                            if test_key in tests_seen:
                                raise ValueError(f"Repeated target test: {test_key}")
                            tests_seen.add(test_key)
                            n_test += 1
                            recreated = sat.legacy.spans_from_probabilities(
                                raw, probabilities, selection["M3_threshold"])
                            if recreated != originals[row["document_id"]]:
                                raise ValueError(f"M3 cache does not reproduce recorded test: "
                                                 f"{target}/{row['document_id']}")
                        for threshold in THRESHOLDS:
                            counts, candidate_set, _ = document_counts(row, probabilities, threshold)
                            if partition == "target_test" and threshold == selection["M3_threshold"]:
                                prefix = np.empty(len(raw) + 1, dtype=np.int32)
                                prefix[0] = 0
                                prefix[1:] = np.cumsum(np.fromiter(
                                    (not c.isspace() for c in raw), dtype=np.int8, count=len(raw)))
                                reconstructed = boundaries_from_spans(
                                    raw, originals[row["document_id"]], prefix)
                                if reconstructed != candidate_set:
                                    raise ValueError(f"Candidate conversion differs from M3: "
                                                     f"{target}/{row['document_id']}")
                                exact_recall = sat.score(row, originals[row["document_id"]])["span_tp"]
                                if counts["available_spans"] < exact_recall:
                                    raise ValueError("Candidate upper bound below actual exact-span recall")
                            key = (partition, target, row["corpus"], threshold)
                            by_corpus[key].update(counts)
                            writer.writerow({"partition": partition, "target": target,
                                             "corpus": row["corpus"],
                                             "document_id": row["document_id"],
                                             "threshold": threshold, **counts})
    if n_test != 2690:
        raise ValueError(f"Expected 2,690 unique target tests; found {n_test}")
    detail_tmp.replace(detail_path)
    per_corpus = []
    for (partition, target, corpus, threshold), counts in sorted(by_corpus.items()):
        per_corpus.append({"partition": partition, "target": target, "corpus": corpus,
                           "threshold": threshold, **{key: counts[key] for key in INPUT_FIELDS},
                           **ratios(counts)})
    write_csv(out / "per_corpus.csv", per_corpus, RESULT_FIELDS)

    summary = []
    for partition in ("source_validation", "target_test"):
        for target in sat.CORPORA:
            for threshold in THRESHOLDS:
                subset = [row for row in per_corpus if row["partition"] == partition
                          and row["target"] == target and row["threshold"] == threshold]
                expected = 5 if partition == "source_validation" else 1
                if len(subset) != expected:
                    raise ValueError(f"Incomplete {partition} summary for {target}/{threshold}")
                average = lambda key: sum(row[key] for row in subset if row[key] is not None) / len(
                    [row for row in subset if row[key] is not None]) if any(
                        row[key] is not None for row in subset) else None
                summary.append({"partition": partition, "target": target,
                                "threshold": threshold, "n_corpora": len(subset),
                                "documents": sum(row["documents"] for row in subset),
                                "macro_boundary_recall": average("boundary_recall"),
                                "macro_span_upper_recall": average("span_upper_recall"),
                                "macro_gap_span_upper_recall": average("gap_span_upper_recall"),
                                "macro_mean_candidates_per_document": average(
                                    "mean_candidates_per_document"),
                                "macro_mean_score_candidates_per_document": average(
                                    "mean_score_candidates_per_document"),
                                "macro_mean_forced_additions_per_document": average(
                                    "mean_forced_additions_per_document")})
            curve = [row for row in summary if row["partition"] == partition
                     and row["target"] == target]
            if any(high["macro_span_upper_recall"] > low["macro_span_upper_recall"] + 1e-12 or
                   high["macro_mean_candidates_per_document"] >
                   low["macro_mean_candidates_per_document"] + 1e-12
                   for low, high in zip(curve, curve[1:])):
                raise ValueError(f"Non-monotone candidate curve: {partition}/{target}")
    write_csv(out / "per_rotation.csv", summary, MACRO_FIELDS)
    macro_test = []
    for threshold in THRESHOLDS:
        subset = [row for row in summary if row["partition"] == "target_test"
                  and row["threshold"] == threshold]
        if len(subset) != 6:
            raise ValueError(f"Missing target test for threshold {threshold}")
        macro_test.append({"partition": "target_test", "target": "six_domain_macro",
                           "threshold": threshold, "n_corpora": 6,
                           "documents": sum(row["documents"] for row in subset),
                           **{field: sum(row[field] for row in subset) / 6 for field in
                              MACRO_FIELDS[5:] if all(row[field] is not None for row in subset)}})
    write_csv(out / "macro_test.csv", macro_test, MACRO_FIELDS)
    write_plots(summary, macro_test, selections, out)
    write_report(summary, macro_test, selections, out)
    output_files = ("per_document.csv", "per_corpus.csv", "per_rotation.csv",
                    "macro_test.csv", "tradeoff_by_rotation.svg", "tradeoff_by_rotation.png",
                    "tradeoff_macro_test.svg", "tradeoff_macro_test.png", "RESULTS.md")
    result_manifest = {
        "experiment": "rq1_candidate_boundaries_v1",
        "reference": manifest["reference"],
        "gold_snapshot_sha256": manifest["gold_snapshot_sha256"],
        "common_split_sha256": manifest["common_split_sha256"],
        "thresholds": list(THRESHOLDS),
        "comparison": "probability > threshold, matching the existing SaT conversion",
        "boundary_convention": "unique nonspace-character ranks; trivial document bounds excluded",
        "span_convention": "both trimmed gold endpoints among candidates plus document bounds",
        "forced_boundaries": "newlines; counted separately from score candidates",
        "interpretation": "Candidate exact-span recall is only an upper bound for a future refiner.",
        "source_validation_is_model_selection_data": True,
        "target_test_used_for_selection": False,
        "n_unique_target_test_documents": n_test,
        "selections": selections,
        "input_sha256": input_hashes,
        "script_sha256": digest(Path(__file__)),
        "output_sha256": {name: digest(out / name) for name in output_files},
    }
    temporary = out / "manifest.json.tmp"
    temporary.write_text(json.dumps(result_manifest, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(out / "manifest.json")
    print(json.dumps({"output": str(out), "test_documents": n_test,
                      "macro_test": [{"threshold": row["threshold"],
                                      "mean_candidates": row["macro_mean_candidates_per_document"],
                                      "span_upper_recall": row["macro_span_upper_recall"]}
                                     for row in macro_test]}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("run", help="Analyze retained local M3 probability caches")
    command.add_argument("--bundle", type=Path,
                         default=ROOT / "annotation" / "rq1_final_v1" / "bundle")
    command.add_argument("--runs", type=Path, default=ROOT / "rq1_segmentation_final_v1")
    command.add_argument("--output", type=Path,
                         default=ROOT / "annotation" / "rq1_final_v1" / "candidate_boundaries_v1")
    command.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
