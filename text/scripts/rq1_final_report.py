#!/usr/bin/env python3
"""Rescore the shared narrative_v3 tests for the publication RQ1 campaign.

Reads only fixed gold, pinned probability caches and source-selected thresholds.
No test label is used for checkpoint or threshold selection.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path

import numpy as np

import rq1_final as campaign
import sat_lodo_gold_v3 as sat
import sat_lodo_checkpoint_select as checkpoint

ROOT = Path(__file__).resolve().parents[2]
PYSBD = ROOT / "annotation" / "rq1_pysbd_v3_20261017"
SINGLE = campaign.SINGLE_RUNS
LODO = ROOT / "safer_sat_rq1_final" / "runs" / campaign.NAME
OUTPUT = ROOT / "annotation" / "rq1_final_v1" / "results"
BASELINE_OUTPUT = ROOT / "annotation" / "rq1_final_v1" / "local_baselines"
MODELS = ("M0", "M1", "M2", "M3")
ROLES = sat.ROLES
COUNTS = ("span_tp", "span_pred", "span_ref", "boundary_tp", "boundary_pred",
          "boundary_ref", "gold_units", "pred_segments", "fusion", "fragmented",
          "merge_same_role", "merge_cross_role", "boundary_shift", "missed", "spurious")


def normalized(raw: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    output = [sat.trim_span(raw, start, end) for start, end in spans]
    output = [(start, end) for start, end in output if start < end]
    if output != sorted(set(output)) or any(a[1] > b[0] for a, b in zip(output, output[1:])):
        raise ValueError("Invalid normalized spans")
    return output


def iou(pred: tuple[int, int], gold: tuple[int, int]) -> float:
    overlap = max(0, min(pred[1], gold[1]) - max(pred[0], gold[0]))
    return overlap / (max(pred[1], gold[1]) - min(pred[0], gold[0])) if overlap else 0.0


def soft_credit(pred: list[tuple[int, int]], gold: list[tuple[int, int]],
                minimum: float = 0.0) -> float:
    """Optimal ordered one-to-one IoU matching for disjoint span sequences."""
    previous = [0.0] * (len(gold) + 1)
    for item in pred:
        current = [0.0]
        for index, ref in enumerate(gold, start=1):
            weight = iou(item, ref)
            current.append(max(previous[index], current[-1],
                               previous[index - 1] + weight if weight > 0 and
                               weight >= minimum else 0.0))
        previous = current
    return previous[-1]


def score(row: dict, spans: list[tuple[int, int]]) -> dict:
    raw = row["raw_text"]
    gold = normalized(raw, [(u["start"], u["end"]) for u in row["units"]])
    pred = normalized(raw, spans)
    counts = dict(sat.score(row, spans, details=True))
    credit = soft_credit(pred, gold)
    credit75 = soft_credit(pred, gold, 0.75)
    counts.update({"soft_credit": credit, "soft75_credit": credit75,
                   "fusion": 0,
                   "merge_same_role": 0, "merge_cross_role": 0,
                   "boundary_shift": 0, "missed": 0, "spurious": 0})
    roles = [(sat.trim_span(raw, u["start"], u["end"]), u["role"]) for u in row["units"]]
    for item in pred:
        overlapped_roles = [role for ref, role in roles if iou(item, ref) > 0]
        if len(overlapped_roles) >= 2:
            counts["fusion"] += 1
            if len(set(overlapped_roles)) == 1:
                counts["merge_same_role"] += 1
            else:
                counts["merge_cross_role"] += 1
        if not any(iou(item, ref) >= 0.75 for ref in gold):
            counts["spurious"] += 1
    for item in gold:
        overlapping = [p for p in pred if iou(p, item) > 0]
        if len(overlapping) == 1 and item != overlapping[0] and iou(overlapping[0], item) >= 0.75:
            counts["boundary_shift"] += 1
        if not any(iou(p, item) >= 0.75 for p in pred):
            counts["missed"] += 1
    return counts


def checked_spans(row: dict, records: list) -> list[tuple[int, int]]:
    raw = row["raw_text"]
    spans = []
    cursor = 0
    for item in records:
        start, end = item["start"], item["end"]
        if (type(start) is not int or type(end) is not int or
                not 0 <= cursor <= start < end <= len(raw) or
                item.get("text") != raw[start:end]):
            raise ValueError(f"Invalid predicted offset: {row['corpus']}/{row['document_id']}")
        spans.append((start, end))
        cursor = end
    return spans


def prediction_file(path: Path, expected: list[dict]) -> dict[str, list[tuple[int, int]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    by_id = {row["document_id"]: row for row in expected}
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            document_id = item["document_id"]
            if document_id not in by_id or document_id in result:
                raise ValueError(f"Unexpected or repeated prediction: {path}/{document_id}")
            result[document_id] = checked_spans(by_id[document_id], item["segments"])
    if set(result) != set(by_id):
        raise ValueError(f"Missing predictions: {path}")
    return result


def write_predictions(path: Path, rows: list[dict], spans: dict[str, list[tuple[int, int]]],
                      threshold: float | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            raw = row["raw_text"]
            stream.write(json.dumps({"corpus": row["corpus"], "document_id": row["document_id"],
                "threshold": threshold,
                "segments": [{"start": a, "end": b, "text": raw[a:b]}
                             for a, b in spans[row["document_id"]]]}, ensure_ascii=False) + "\n")
    temporary.replace(path)


def pysbd_predictions(db: sqlite3.Connection, rows: list[dict]) -> dict:
    result = {}
    for row in rows:
        found = db.execute("SELECT text_sha256,spans_json FROM predictions WHERE corpus=? AND document_id=?",
                           (row["corpus"], row["document_id"])).fetchone()
        if found is None or found[0] != hashlib.sha256(row["raw_text"].encode("utf-8")).hexdigest():
            raise ValueError(f"PySBD cache mismatch: {row['corpus']}/{row['document_id']}")
        result[row["document_id"]] = [(int(a), int(b)) for a, b in json.loads(found[1])]
    return result


def probability_predictions(db: sqlite3.Connection, rows: list[dict], threshold: float) -> dict:
    return {row["document_id"]: sat.legacy.spans_from_probabilities(
        row["raw_text"], sat.get_probs(db, row), threshold) for row in rows}


def metric_rows(rows: list[dict], pred: dict, experiment: str, target: str,
                model: str, source: str = "") -> list[dict]:
    result = []
    for row in rows:
        result.append({"experiment": experiment, "source": source, "target": target,
                       "corpus": row["corpus"], "document_id": row["document_id"],
                       "model": model, **score(row, pred[row["document_id"]])})
    return result


def aggregate(records: list[dict], *, experiment: str, source: str, target: str, model: str) -> dict:
    if not records:
        raise ValueError("Cannot score empty target")
    totals = collections.Counter()
    for row in records:
        totals.update({key: row.get(key, 0) for key in COUNTS + ("soft_credit", "soft75_credit")})
        for role in ROLES:
            totals[f"role_{role}_tp"] += row.get(f"role_{role}_tp", 0)
            totals[f"role_{role}_ref"] += row.get(f"role_{role}_ref", 0)
    sp, sr, sf = sat.prf(totals["span_tp"], totals["span_pred"], totals["span_ref"])
    bp, br, bf = sat.prf(totals["boundary_tp"], totals["boundary_pred"], totals["boundary_ref"])
    return {"experiment": experiment, "source": source, "target": target, "model": model,
            "n_test": len(records), **{key: totals[key] for key in COUNTS},
            "exact_precision": sp, "exact_recall": sr, "exact_f1": sf,
            "boundary_precision": bp, "boundary_recall": br, "boundary_f1": bf,
            "soft_precision": totals["soft_credit"] / totals["span_pred"] if totals["span_pred"] else 0.0,
            "soft_recall": totals["soft_credit"] / totals["span_ref"] if totals["span_ref"] else 0.0,
            "soft_f1": 2 * totals["soft_credit"] / (totals["span_pred"] + totals["span_ref"])
            if totals["span_pred"] + totals["span_ref"] else 0.0,
            "soft75_f1": 2 * totals["soft75_credit"] / (totals["span_pred"] + totals["span_ref"])
            if totals["span_pred"] + totals["span_ref"] else 0.0,
            **{f"role_{role}_ref": totals[f"role_{role}_ref"] for role in ROLES},
            **{f"role_{role}_exact_recall":
               totals[f"role_{role}_tp"] / totals[f"role_{role}_ref"]
               if totals[f"role_{role}_ref"] else 0.0 for role in ROLES}}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap(archive: dict, reps: int = 2000) -> list[dict]:
    rng = np.random.default_rng(20261018)
    output = []
    macro = collections.defaultdict(list)
    for target in sat.CORPORA:
        records = {model: archive[target, model] for model in MODELS}
        document_ids = [r["document_id"] for r in records["M0"]]
        if any([r["document_id"] for r in records[model]] != document_ids for model in MODELS):
            raise ValueError(f"Bootstrap data are not paired: {target}")
        n = len(document_ids)
        sample = rng.integers(0, n, size=(reps, n))
        sampled_f1 = {}
        observed_f1 = {}
        for model, rows in records.items():
            counts = np.asarray([[r["span_tp"], r["span_pred"], r["span_ref"]]
                                 for r in rows], dtype=np.int64)
            summed = counts[sample].sum(axis=1)
            sampled_f1[model] = np.divide(
                2 * summed[:, 0], summed[:, 1] + summed[:, 2], out=np.zeros(reps),
                where=(summed[:, 1] + summed[:, 2]) != 0)
            total = counts.sum(axis=0)
            observed_f1[model] = (2 * total[0] / (total[1] + total[2])
                                  if total[1] + total[2] else 0.0)
        for left, right in zip(MODELS, MODELS[1:]):
            comparison = f"{right}_minus_{left}"
            delta = sampled_f1[right] - sampled_f1[left]
            macro[comparison].append(delta)
            output.append({"target": target, "comparison": comparison,
                           "bootstrap_reps": reps,
                           "delta_exact_f1": float(observed_f1[right] - observed_f1[left]),
                           "ci95_low": float(np.quantile(delta, 0.025)),
                           "ci95_high": float(np.quantile(delta, 0.975))})
    for comparison, samples in macro.items():
        averaged = np.mean(np.stack(samples), axis=0)
        output.append({"target": "MACRO_6", "comparison": comparison,
                       "bootstrap_reps": reps,
                       "delta_exact_f1": float(np.mean([
                           row["delta_exact_f1"] for row in output
                           if row["comparison"] == comparison])),
                       "ci95_low": float(np.quantile(averaged, 0.025)),
                       "ci95_high": float(np.quantile(averaged, 0.975))})
    return output


def matrix_rows(metrics: dict, model: str) -> list[dict]:
    return [{"source": source, **{target: ("N/A" if source == target else
             metrics["single_source", source, target, model]["exact_f1"])
             for target in sat.CORPORA}} for source in sat.CORPORA]


def heatmap(path: Path, matrix: list[dict], title: str) -> None:
    width, height = 1020, 540
    cellw, cellh = 125, 55
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;font-size:13px;fill:#17283a}</style>',
             f'<text x="35" y="35" style="font-size:19px;font-weight:bold">{title}</text>']
    labels = {"btp": "BTP", "metallurgie": "Métallurgie", "caou_chimie_plas": "Chimie",
              "bois_textille": "Bois-textile", "service_temporaire": "Service temp.",
              "nicollin": "Nicollin"}
    for j, target in enumerate(sat.CORPORA):
        parts.append(f'<text x="{205+j*cellw+cellw/2:.0f}" y="92" text-anchor="middle">{labels[target]}</text>')
    for i, row in enumerate(matrix):
        y = 108 + i * cellh
        parts.append(f'<text x="185" y="{y+32}" text-anchor="end">{labels[row["source"]]}</text>')
        for j, target in enumerate(sat.CORPORA):
            value = row[target]
            x = 205 + j * cellw
            if value == "N/A":
                fill, label = "#e9edf1", "N/A"
            else:
                saturation = max(0, min(255, round(235 - 145 * float(value))))
                fill, label = f'rgb({saturation},{saturation+10 if saturation < 245 else 255},255)', f"{value:.3f}"
            parts.append(f'<rect x="{x}" y="{y}" width="{cellw-4}" height="{cellh-4}" fill="{fill}"/>')
            parts.append(f'<text x="{x+cellw/2:.0f}" y="{y+31}" text-anchor="middle">{label}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def role_figure(path: Path, rows: list[dict]) -> None:
    """Macro recall by gold role; the segmenter never predicts roles."""
    by_role = {(model, role): float(np.mean([r["exact_recall"] for r in rows
                if r["model"] == model and r["role"] == role]))
               for model in ("M2", "M3") for role in ROLES}
    labels = {"AUTRE": "OTHER"}
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="950" height="470" viewBox="0 0 950 470">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#17283a}</style>',
             '<text x="25" y="35" font-size="20">Rappel exact par rôle GOLD — LODO</text>',
             '<text x="25" y="58" font-size="12">Moyenne macro des six domaines, rôles non prédits par SaT</text>']
    for i, role in enumerate(ROLES):
        x = 80 + i * 170
        parts.append(f'<text x="{x+55}" y="420" text-anchor="middle">{labels.get(role, role)}</text>')
        for j, model in enumerate(("M2", "M3")):
            value = by_role[model, role]
            height = value * 320
            color = "#5481aa" if model == "M2" else "#dd8545"
            parts.append(f'<rect x="{x+j*55}" y="{390-height:.1f}" width="46" height="{height:.1f}" fill="{color}"/>')
            parts.append(f'<text x="{x+j*55+23}" y="{380-height:.1f}" text-anchor="middle" font-size="11">{value:.2f}</text>')
    parts += ['<rect x="780" y="70" width="14" height="14" fill="#5481aa"/>',
              '<text x="800" y="82" font-size="12">M2 calibré</text>',
              '<rect x="780" y="94" width="14" height="14" fill="#dd8545"/>',
              '<text x="800" y="106" font-size="12">M3 LoRA</text>', '</svg>']
    path.write_text("\n".join(parts), encoding="utf-8")


def gain_figure(path: Path, rows: list[dict]) -> None:
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="450" viewBox="0 0 1000 450">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#17283a}</style>',
             '<text x="25" y="35" font-size="20">Gain LODO M3 − M2 en F1 exact</text>',
             '<text x="25" y="57" font-size="12">IC à 95 % par bootstrap apparié de récits (2 000 répétitions)</text>']
    xzero, scale = 510, 900
    parts.append(f'<line x1="{xzero}" y1="85" x2="{xzero}" y2="425" stroke="#65717e"/>')
    for i, row in enumerate(rows):
        y = 95 + i * 48
        low, high = xzero + row["ci95_low"] * scale, xzero + row["ci95_high"] * scale
        dot = xzero + row["delta_exact_f1"] * scale
        parts.append(f'<text x="230" y="{y+5}" text-anchor="end" font-size="12">{row["target"]}</text>')
        parts.append(f'<line x1="{low:.1f}" y1="{y}" x2="{high:.1f}" y2="{y}" stroke="#496887" stroke-width="2"/>')
        parts.append(f'<circle cx="{dot:.1f}" cy="{y}" r="5" fill="#d67b3d"/>')
        parts.append(f'<text x="920" y="{y+5}" font-size="11">{row["delta_exact_f1"]:+.3f}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def latex_table(path: Path, rows: list[dict]) -> None:
    def cell(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value).replace("_", "\\_").replace("%", "\\%")
    lines = ["\\begin{tabular}{lrrrrr}", "\\toprule",
             "Domain & PySBD & SaT & SaT cal. & SaT LoRA & Gain \\\\", "\\midrule"]
    for row in rows:
        lines.append(" & ".join(cell(row[key]) for key in
                     ("target", "M0_exact_f1", "M1_exact_f1", "M2_exact_f1",
                      "M3_exact_f1", "gain_M3_minus_M2")) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def baselines(args: argparse.Namespace) -> None:
    """Complete M0/M1 and source-only M2 before any LODO GPU run."""
    from tqdm import tqdm

    bundle, output = args.bundle.resolve(), args.baseline_output.resolve()
    manifest = campaign.check_bundle(bundle)
    output.mkdir(parents=True, exist_ok=True)
    metrics = []
    choices = []
    with sqlite3.connect(bundle / "base_probabilities.sqlite3") as base_db, \
         sqlite3.connect(args.pysbd / "cache.sqlite3") as pysbd_db:
        for target in tqdm(sat.CORPORA, desc="Baselines RQ1", unit="corpus"):
            folder = bundle / "folds" / f"target_{target}"
            validation = sat.read_jsonl(folder / "val.jsonl")
            curve = checkpoint.validation_curve(validation, base_db, "base", -1, -1.0)
            chosen = checkpoint.best_row(curve)
            choices.append({"target": target, "M2_threshold": chosen["threshold"],
                            "validation_macro_exact_f1": chosen["source_macro_span_f1"],
                            "n_validation": len(validation)})
            test = sat.read_jsonl(folder / "test.jsonl")
            variants = {"M0": (pysbd_predictions(pysbd_db, test), None),
                        "M1": (probability_predictions(base_db, test, sat.DEFAULT), sat.DEFAULT),
                        "M2": (probability_predictions(base_db, test, chosen["threshold"]),
                               chosen["threshold"])}
            for model, (pred, threshold) in variants.items():
                write_predictions(output / "predictions" / target / f"{model}.jsonl",
                                  test, pred, threshold)
                doc_rows = metric_rows(test, pred, "lodo", target, model)
                metrics.append(aggregate(doc_rows, experiment="lodo", source="",
                                         target=target, model=model))
    write_csv(output / "baseline_metrics.csv", metrics)
    write_csv(output / "selected_source_thresholds.csv", choices)
    campaign.json_file(output / "manifest.json", {
        "campaign": campaign.NAME, "gold_snapshot_sha256": manifest["gold_snapshot_sha256"],
        "common_split_sha256": manifest["common_split_sha256"],
        "selection": "M2: macro exact F1 on the five source validations only",
        "n_test": sum(campaign.EXPECTED_COUNTS[c][1] for c in sat.CORPORA),
        "thresholds": choices,
        "files": {name: sat.digest(output / name) for name in
                  ("baseline_metrics.csv", "selected_source_thresholds.csv")}})
    print(json.dumps({"baseline_output": str(output), "thresholds": choices},
                     ensure_ascii=False, indent=2))


def report(args: argparse.Namespace) -> None:
    bundle, single_root, lodo_root, output = (args.bundle.resolve(), args.single.resolve(),
                                              args.lodo.resolve(), args.output.resolve())
    manifest = campaign.check_bundle(bundle)
    campaign.audit_single(argparse.Namespace(source=args.single_bundle, runs=single_root,
                                              output=None))
    complete = json.loads((lodo_root / "cluster_complete.json").read_text(encoding="utf-8"))
    if (complete["campaign"] != campaign.NAME or
            complete["gold_snapshot_sha256"] != manifest["gold_snapshot_sha256"] or
            complete["common_split_sha256"] != manifest["common_split_sha256"] or
            complete["model"] != sat.MODEL or
            {row["target"] for row in complete["results"]} != set(sat.CORPORA) or
            len(complete["results"]) != 6):
        raise ValueError("Cluster results do not match the fixed local tests")
    transfer_path = lodo_root / "transfer_manifest.json"
    if not complete.get("fixture_only"):
        if not transfer_path.is_file():
            raise ValueError("Missing verified cluster transfer manifest")
        transfer = json.loads(transfer_path.read_text(encoding="utf-8"))
        if (transfer["campaign"] != campaign.NAME or
                transfer["gold_snapshot_sha256"] != manifest["gold_snapshot_sha256"] or
                transfer["common_split_sha256"] != manifest["common_split_sha256"]):
            raise ValueError("Cluster transfer belongs to another reference")
        for relative, expected in transfer["files"].items():
            member = (lodo_root / relative).resolve()
            if (not member.is_relative_to(lodo_root) or not member.is_file() or
                    sat.digest(member) != expected):
                raise ValueError(f"Missing or changed cluster transfer member: {relative}")
    output.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    detailed = []
    archive = {}
    selections = []
    with ExitStack() as stack:
        base_db = stack.enter_context(closing(sqlite3.connect(bundle / "base_probabilities.sqlite3")))
        pysbd_db = stack.enter_context(closing(sqlite3.connect(args.pysbd / "cache.sqlite3")))
        source_db = {source: stack.enter_context(closing(sqlite3.connect(
            single_root / f"source_{source}" / "heldout_probabilities.sqlite3")))
            for source in sat.CORPORA}
        for target in sat.CORPORA:
            test = sat.read_jsonl(bundle / "folds" / f"target_{target}" / "test.jsonl")
            cluster_run = lodo_root / f"target_{target}"
            result = json.loads((cluster_run / "metrics.json").read_text(encoding="utf-8"))
            selection = json.loads((cluster_run / "selection.json").read_text(encoding="utf-8"))
            if (result["target"] != target or result["n_test"] != len(test) or
                    result["selection_sha256"] != sat.digest(cluster_run / "selection.json") or
                    result["selection"] != selection):
                raise ValueError(f"LODO output is incomplete: {target}")
            if not complete.get("fixture_only") and result["bundle"]["bundle_manifest_sha256"] != complete["bundle_manifest_sha256"]:
                raise ValueError(f"LODO output belongs to another bundle: {target}")
            if not complete.get("fixture_only"):
                for filename, expected in selection["M3_weights"].items():
                    if sat.digest(cluster_run / "selected_adapter" / filename) != expected:
                        raise ValueError(f"Selected LoRA weights differ from evaluation: {target}")
            local_baseline = args.baseline_output / "manifest.json"
            if local_baseline.is_file():
                existing = json.loads(local_baseline.read_text(encoding="utf-8"))
                if (existing["gold_snapshot_sha256"] != manifest["gold_snapshot_sha256"] or
                        existing["common_split_sha256"] != manifest["common_split_sha256"]):
                    raise ValueError("Local baseline belongs to another reference")
                old_threshold = next(r["M2_threshold"] for r in existing["thresholds"]
                                     if r["target"] == target)
                if old_threshold != selection["M2_threshold"]:
                    raise ValueError(f"Cluster and local M2 calibration differ: {target}")
            selections.append({"experiment": "lodo", "source": "five_domains", "target": target,
                               "M2_threshold": selection["M2_threshold"],
                               "M3_checkpoint": selection["M3_candidate"],
                               "M3_threshold": selection["M3_threshold"]})
            fixed = {
                "M0": (pysbd_predictions(pysbd_db, test), None),
                "M1": (probability_predictions(base_db, test, sat.DEFAULT), sat.DEFAULT),
                "M2": (probability_predictions(base_db, test, selection["M2_threshold"]),
                       selection["M2_threshold"]),
            }
            for model in ("M1", "M2", "M3"):
                path = cluster_run / "predictions" / f"{model}.jsonl"
                if result["prediction_sha256"][model] != sat.digest(path):
                    raise ValueError(f"Changed cluster prediction: {path}")
                observed = prediction_file(path, test)
                if model in fixed and observed != fixed[model][0]:
                    raise ValueError(f"Local and cluster baseline disagree: {target}/{model}")
                if model == "M3":
                    fixed[model] = (observed, selection["M3_threshold"])
            for model, (pred, threshold) in fixed.items():
                write_predictions(output / "predictions" / "lodo" / target / f"{model}.jsonl",
                                  test, pred, threshold)
                doc_rows = metric_rows(test, pred, "lodo", target, model)
                archive[target, model] = doc_rows
                detailed.extend(doc_rows)
                all_metrics.append(aggregate(doc_rows, experiment="lodo", source="",
                                             target=target, model=model))
            for source in sat.CORPORA:
                if source == target:
                    continue
                source_run = single_root / f"source_{source}"
                selected = json.loads((source_run / "selection.json").read_text(encoding="utf-8"))
                source_metrics = json.loads((source_run / "metrics.json").read_text(encoding="utf-8"))
                selections.append({"experiment": "single_source", "source": source, "target": target,
                                   "M2_threshold": selected["M1_threshold"],
                                   "M3_checkpoint": selected["M2_candidate"],
                                   "M3_threshold": selected["M2_threshold"]})
                for model, threshold, db in (
                        ("M2", selected["M1_threshold"], base_db),
                        ("M3", selected["M2_threshold"], source_db[source])):
                    pred = probability_predictions(db, test, threshold)
                    doc_rows = metric_rows(test, pred, "single_source", target, model, source)
                    old_label = "M1" if model == "M2" else "M2"
                    if not math.isclose(aggregate(doc_rows, experiment="single_source", source=source,
                                                  target=target, model=model)["exact_f1"],
                                        source_metrics["test_sets"][target][old_label]["span_f1"],
                                        rel_tol=0, abs_tol=1e-12):
                        raise ValueError(f"Old single-source exact score was not reproduced: {source}->{target}")
                    write_predictions(output / "predictions" / "single_source" / source /
                                      target / f"{model}.jsonl", test, pred, threshold)
                    detailed.extend(doc_rows)
                    all_metrics.append(aggregate(doc_rows, experiment="single_source",
                                                 source=source, target=target, model=model))
    by_key = {(row["experiment"], row["source"], row["target"], row["model"]): row
              for row in all_metrics}
    if len(by_key) != 84:  # 24 LODO + 60 off-diagonal single-source metrics
        raise ValueError(f"Incomplete model/domain score grid: {len(by_key)}")
    main = []
    for target in sat.CORPORA:
        lookup = {model: by_key["lodo", "", target, model] for model in MODELS}
        main.append({"target": target, "n_test": lookup["M0"]["n_test"],
                     **{f"{model}_{key}": lookup[model][key] for model in MODELS
                        for key in ("exact_precision", "exact_recall", "exact_f1",
                                    "boundary_precision", "boundary_recall", "boundary_f1",
                                    "soft_f1", "soft75_f1")},
                     "gain_M3_minus_M2": lookup["M3"]["exact_f1"] - lookup["M2"]["exact_f1"]})
    macro = {"target": "MACRO_6", "n_test": sum(row["n_test"] for row in main)}
    for key in main[0]:
        if key not in ("target", "n_test"):
            macro[key] = sum(float(row[key]) for row in main) / 6
    main.append(macro)
    matrix = matrix_rows(by_key, "M3")
    calibrated_matrix = matrix_rows(by_key, "M2")
    comparison = []
    for target in sat.CORPORA:
        transfers = [by_key["single_source", source, target, "M3"]["exact_f1"]
                     for source in sat.CORPORA if source != target]
        comparison.append({"target": target, "n_test": campaign.EXPECTED_COUNTS[target][1],
                           "lodo_M3_exact_f1": by_key["lodo", "", target, "M3"]["exact_f1"],
                           "single_source_mean": float(np.mean(transfers)),
                           "single_source_sd": float(np.std(transfers, ddof=1)),
                           "single_source_oracle_max": max(transfers)})
    roles = [{"target": target, "model": model, "role": role,
              "gold_support": by_key["lodo", "", target, model][f"role_{role}_ref"],
              "exact_recall": by_key["lodo", "", target, model][f"role_{role}_exact_recall"]}
             for target in sat.CORPORA for model in ("M2", "M3") for role in ROLES]
    ci = bootstrap(archive)
    for name, rows in (("table_A_lodo.csv", main), ("table_B_transfer_M3.csv", matrix),
                       ("annex_transfer_M2.csv", calibrated_matrix),
                       ("table_C_single_vs_multi.csv", comparison),
                       ("annex_all_metrics.csv", all_metrics),
                       ("annex_role_recall.csv", roles),
                       ("annex_selections.csv", selections),
                       ("annex_document_metrics.csv", detailed),
                       ("bootstrap_paired_2000.csv", ci)):
        write_csv(output / name, rows)
    heatmap(output / "figure_transfer_M3.svg", matrix, "Transfert mono-source : F1 exact")
    role_figure(output / "figure_role_recall.svg", roles)
    gain_figure(output / "figure_lodo_gain_ci.svg",
                [row for row in ci if row["comparison"] == "M3_minus_M2"])
    latex_table(output / "table_A_lodo.tex", main)
    campaign.json_file(output / "report_manifest.json", {
        "campaign": campaign.NAME, "gold_snapshot_sha256": manifest["gold_snapshot_sha256"],
        "common_split_sha256": manifest["common_split_sha256"],
        "cluster_complete_sha256": sat.digest(lodo_root / "cluster_complete.json"),
        "single_source_bundle_sha256": sat.digest(args.single_bundle / "manifest.json"),
        "method": "exact and boundary F1: micro within each domain, macro over six domains; "
                  "Soft F1: character IoU, maximum-weight one-to-one matching, all positive overlaps; "
                  "IoU >= .75 is sensitivity only; paired bootstrap: 2000 narratives, one seed.",
        "reference_limit": "Researcher-confirmed AI preannotations; no independent double annotation.",
        "training_limit": "Official trainer reconstructs factual units; evaluation uses raw narratives.",
        "tables": ["table_A_lodo.csv", "table_B_transfer_M3.csv", "table_C_single_vs_multi.csv"]})
    gains = [row["gain_M3_minus_M2"] for row in main[:-1]]
    (output / "RESULTS.md").write_text(
        "# RQ1 — segmentation factuelle\n\n"
        "Ce rapport utilise exclusivement le GOLD narrative_v3 et les 2 690 récits de test communs. "
        "Le GOLD a été confirmé par la chercheuse à partir de préannotations IA ; il ne s'agit pas "
        "d'une double annotation indépendante.\n\n"
        "Les tableaux CSV et LaTeX et la matrice SVG sont calculés par `rq1_final_report.py`. "
        "M0=PySBD, M1=SaT préentraîné, M2=SaT calibré sur sources, M3=SaT+LoRA. "
        "La diagonale du transfert mono-source est N/A. Le maximum mono-source est un oracle "
        "rétrospectif, non une stratégie de sélection. Une seule graine LoRA a été utilisée ; "
        "les IC bootstrap mesurent la variabilité des récits, pas celle des graines.\n\n"
        f"F1 exact macro : PySBD {macro['M0_exact_f1']:.3f}, SaT initial "
        f"{macro['M1_exact_f1']:.3f}, SaT calibré {macro['M2_exact_f1']:.3f}, "
        f"SaT + LoRA {macro['M3_exact_f1']:.3f}. Gain M3−M2 : "
        f"{macro['gain_M3_minus_M2']:+.3f}, positif dans {sum(x > 0 for x in gains)}/6 "
        "domaines. Les différences par domaine et les IC figurent dans les tableaux joints. "
        "Ces résultats n'isolent pas l'effet causal de la diversité des sources, car les volumes "
        "d'entraînement diffèrent.\n",
        encoding="utf-8")
    print(json.dumps({"output": str(output), "n_test": 2690, "macro_exact":
                      {model: macro[f"{model}_exact_f1"] for model in MODELS}},
                     ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=campaign.OUTPUT)
    parser.add_argument("--single-bundle", type=Path, default=campaign.SOURCE)
    parser.add_argument("--single", type=Path, default=SINGLE)
    parser.add_argument("--pysbd", type=Path, default=PYSBD)
    parser.add_argument("--lodo", type=Path, default=LODO)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--baseline-output", type=Path, default=BASELINE_OUTPUT)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    if args.baseline_only:
        baselines(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
