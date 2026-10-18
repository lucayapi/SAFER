#!/usr/bin/env python3
"""Post-hoc overlap scores for the fixed, held-out in-domain SaT experiment.

No checkpoint or threshold is reselected. M0/M1 use the cached base
probabilities and M2 uses the cached probabilities of its selected adapter.
The result is descriptive: the matching IoU cutoffs were specified after the
original exact-span evaluation.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from contextlib import closing
from html import escape
from pathlib import Path

import numpy as np

import sat_lodo_gold_v3 as base

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "sat12l_sm_gold_v3_indomain_v1"
DEFAULT_BUNDLE = ROOT / "safer_sat_indomain" / "bundle" / EXPERIMENT
DEFAULT_RUN = ROOT / "safer_sat_indomain" / "runs" / EXPERIMENT
CUTOFFS = (0.50, 0.75, 0.90)
MODELS = ("M0", "M1", "M2")


def cached_spans(raw: str, probabilities: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Decode like the pinned wtpsplit indices_to_sentences + pieces_to_spans.

    This small local decoder avoids importing the full model stack for a
    CPU-only post-hoc score. Existing exact-span counts must reproduce all
    three original model results before any soft score is saved.
    """
    if len(probabilities) != len(raw):
        raise ValueError("Probability vector length differs from narrative")
    sentences = []
    offset = 0
    last_index = 0
    for index in np.flatnonzero(probabilities > threshold):
        index = int(index) + 1
        while index < len(raw) and raw[index].isspace():
            index += 1
        sentence = raw[offset:index]
        if sentence:
            sentences.append(sentence)
        offset = index
        last_index = index
    if last_index != len(raw):
        final = raw[last_index:]
        if final:
            sentences.append(final)
    pieces = []
    for index, sentence in enumerate(sentences):
        if index < len(sentences) - 1 and sentence.endswith("\n"):
            sentence = sentence[:-1]
        pieces.extend(sentence.split("\n"))
    return base.legacy.pieces_to_spans(raw, pieces)


def normalize(raw: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    trimmed = [base.trim_span(raw, start, end) for start, end in spans]
    result = [(start, end) for start, end in trimmed if start < end]
    if len(result) != len(set(result)) or result != sorted(result):
        raise ValueError("Duplicate or unordered nonempty spans")
    if any(left[1] > right[0] for left, right in zip(result, result[1:])):
        raise ValueError("Overlapping spans")
    return result


def iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    if not intersection:
        return 0.0
    return intersection / (max(left[1], right[1]) - min(left[0], right[0]))


def optimal_credit(pred: list[tuple[int, int]], gold: list[tuple[int, int]],
                   cutoff: float) -> tuple[float, int]:
    """Maximum-weight one-to-one match for two ordered, disjoint span lists.

    Positive-overlap edges cannot cross when both lists are ordered and
    internally nonoverlapping, so sequence alignment is an exact solution.
    """
    previous = [(0.0, 0)] * (len(gold) + 1)
    for predicted in pred:
        current = [(0.0, 0)]
        for index, reference in enumerate(gold, start=1):
            best = max(previous[index], current[-1])
            weight = iou(predicted, reference)
            if weight >= cutoff and weight > 0:
                best = max(best, (previous[index - 1][0] + weight,
                                  previous[index - 1][1] + 1))
            current.append(best)
        previous = current
    return previous[-1]


def metric(credit: float, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    precision = credit / n_pred if n_pred else 0.0
    recall = credit / n_gold if n_gold else 0.0
    f1 = 2 * credit / (n_pred + n_gold) if n_pred + n_gold else 0.0
    return precision, recall, f1


def svg_chart(path: Path, rows: list[dict]) -> None:
    """Small standalone grouped chart, including the original exact scores."""
    width, height = 1050, 530
    left, top, plot_width, plot_height = 200, 75, 800, 355
    grouped = collections.defaultdict(dict)
    for row in rows:
        if row["iou_cutoff"] == 0.75:
            grouped[row["corpus"]][row["model"]] = row
    colors = {"M0": "#58749a", "M1": "#45a0a0", "M2": "#df8c48"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#203047}.title{font-size:20px;font-weight:bold}.label{font-size:13px}.small{font-size:11px}</style>',
             '<text x="25" y="32" class="title">F1 des segments, test in-domain (15 % réservés)</text>',
             '<text x="25" y="52" class="label">Barres : Soft Span F1, IoU ≥ 0,75 · marqueurs : F1 exact</text>']
    for tick in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        x = left + tick * plot_width
        parts.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top + plot_height}" stroke="#e5e9ee"/>')
        parts.append(f'<text x="{x}" y="{top + plot_height + 20}" text-anchor="middle" class="small">{tick:.1f}</text>')
    corpora = list(base.CORPORA)
    row_height = plot_height / len(corpora)
    for j, corpus in enumerate(corpora):
        y = top + j * row_height
        parts.append(f'<text x="{left - 12}" y="{y + 31}" text-anchor="end" class="label">{escape(corpus)}</text>')
        for k, model in enumerate(MODELS):
            item = grouped[corpus][model]
            bar_y = y + 10 + k * 13
            score = item["soft_f1"]
            exact = item["exact_f1"]
            parts.append(f'<rect x="{left}" y="{bar_y}" width="{score * plot_width:.1f}" height="10" fill="{colors[model]}"/>')
            parts.append(f'<line x1="{left + exact * plot_width:.1f}" y1="{bar_y - 2}" x2="{left + exact * plot_width:.1f}" y2="{bar_y + 12}" stroke="#142032" stroke-width="2"/>')
            parts.append(f'<text x="{left + score * plot_width + 5:.1f}" y="{bar_y + 9}" class="small">{score:.3f}</text>')
    legend_y = height - 35
    for k, model in enumerate(MODELS):
        x = left + k * 140
        parts.append(f'<rect x="{x}" y="{legend_y - 10}" width="13" height="10" fill="{colors[model]}"/>')
        parts.append(f'<text x="{x + 20}" y="{legend_y}" class="label">{model}</text>')
    parts.append(f'<line x1="{left + 485}" y1="{legend_y - 13}" x2="{left + 485}" y2="{legend_y + 2}" stroke="#142032" stroke-width="2"/>')
    parts.append(f'<text x="{left + 500}" y="{legend_y}" class="label">F1 exact</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts) + '\n', encoding='utf-8')


def transfer_svg(path: Path, rows: list[dict]) -> None:
    """Side-by-side M2 score and M2-M1 transfer matrices at IoU >= 0.75."""
    labels = {"btp": "BTP", "metallurgie": "Métallurgie",
              "caou_chimie_plas": "Chimie", "bois_textille": "Bois-textile",
              "service_temporaire": "Service temp.", "nicollin": "Nicollin"}
    scores = {(row["source"], row["target"], row["model"]): row["soft_f1"]
              for row in rows if row["iou_cutoff"] == 0.75}
    width, height = 1390, 670
    top, cell_w, cell_h, gap = 145, 85, 68, 105
    lefts = (155, 155 + 6 * cell_w + gap)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#182535}.title{font-size:20px;font-weight:bold}.subtitle{font-size:14px}.label{font-size:12px}.value{font-size:15px;font-weight:600}</style>',
             '<text x="25" y="32" class="title">Transfert de SaT adapté — Soft Span F1 (IoU ≥ 0,75)</text>',
             '<text x="25" y="55" class="subtitle">Lignes : corpus d’entraînement · colonnes : corpus testé · diagonale encadrée : in-domain</text>',
             '<text x="155" y="91" class="subtitle">M2 : F1 de segmentation</text>',
             f'<text x="{lefts[1]}" y="91" class="subtitle">Gain M2 − M1</text>']
    for panel, left in enumerate(lefts):
        for col, target in enumerate(base.CORPORA):
            x = left + col * cell_w + cell_w / 2
            parts.append(f'<text x="{x}" y="{top - 11}" class="label" text-anchor="middle">{escape(labels[target])}</text>')
        for row_index, source in enumerate(base.CORPORA):
            y = top + row_index * cell_h
            parts.append(f'<text x="{left - 8}" y="{y + 41}" class="label" text-anchor="end">{escape(labels[source])}</text>')
            for col, target in enumerate(base.CORPORA):
                m2 = scores[source, target, "M2"]
                gain = m2 - scores[source, target, "M1"]
                value = m2 if panel == 0 else gain
                if panel == 0:
                    intensity = max(0.0, min(1.0, value))
                    color = (round(238 - 153 * intensity), round(246 - 110 * intensity),
                             round(250 - 75 * intensity))
                elif value >= 0:
                    intensity = max(0.0, min(1.0, value / 0.6))
                    color = (round(242 - 160 * intensity), round(248 - 100 * intensity),
                             round(250 - 80 * intensity))
                else:
                    intensity = max(0.0, min(1.0, -value / 0.3))
                    color = (round(250 - 8 * intensity), round(246 - 110 * intensity),
                             round(244 - 120 * intensity))
                x = left + col * cell_w
                stroke = '#152e4d' if source == target else '#dce4eb'
                stroke_width = 3 if source == target else 1
                parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="#{color[0]:02x}{color[1]:02x}{color[2]:02x}" stroke="{stroke}" stroke-width="{stroke_width}"/>')
                shown = f'{value:+.3f}' if panel else f'{value:.3f}'
                parts.append(f'<text x="{x + cell_w / 2}" y="{y + 41}" class="value" text-anchor="middle">{shown}</text>')
    parts.append('<text x="155" y="620" class="subtitle">M1 et M2 : seuils et checkpoints retenus sur la validation du seul corpus source.</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts) + '\n', encoding='utf-8')


def calculate(bundle: Path, run_root: Path, out: Path) -> list[dict]:
    manifest_path = bundle / "manifest.json"
    manifest_hash = base.digest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["reference"] != base.REFERENCE or manifest["fold_order"] != list(base.CORPORA):
        raise ValueError("Unexpected in-domain bundle")
    rows_out: list[dict] = []
    audit: dict[str, dict] = {}
    with closing(base._probability_connection(bundle / "base_probabilities.sqlite3")) as baseline:
        for corpus in base.CORPORA:
            source_run = run_root / f"source_{corpus}"
            selection_path = source_run / "selection.json"
            metrics_path = source_run / "metrics.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            binding = json.loads((source_run / "heldout_probabilities.binding.json").read_text(encoding="utf-8"))
            if (selection["source"] != corpus or metrics["source"] != corpus
                    or selection["bundle_manifest_sha256"] != manifest_hash
                    or metrics["bundle_manifest_sha256"] != manifest_hash
                    or metrics["selection_sha256"] != base.digest(selection_path)
                    or binding["bundle_manifest_sha256"] != manifest_hash
                    or binding["adapter_sha256"] != metrics["adapter_sha256"]
                    or binding["head_sha256"] != metrics["head_sha256"]):
                raise ValueError(f"Selection/model/data provenance mismatch: {corpus}")
            test = base.read_jsonl(bundle / "folds" / f"target_{corpus}" / "test.jsonl")
            if len(test) != manifest["report"][corpus]["split"]["test"]:
                raise ValueError(f"Test count mismatch: {corpus}")
            if len({row["document_id"] for row in test}) != len(test):
                raise ValueError(f"Duplicate test ID: {corpus}")
            aggregate = {model: {cutoff: collections.defaultdict(float) for cutoff in CUTOFFS}
                         for model in MODELS}
            exact = {model: collections.Counter() for model in MODELS}
            thresholds = {"M0": base.DEFAULT, "M1": selection["M1_threshold"],
                          "M2": selection["M2_threshold"]}
            with closing(base._probability_connection(source_run / "heldout_probabilities.sqlite3")) as adapted:
                for row in test:
                    base.validate_gold(row)
                    raw = row["raw_text"]
                    gold = normalize(raw, [(u["start"], u["end"]) for u in row["units"]])
                    probs = {"base": base.get_probs(baseline, row),
                             "adapted": base.get_probs(adapted, row)}
                    for model in MODELS:
                        values = probs["adapted" if model == "M2" else "base"]
                        predicted = cached_spans(raw, values, thresholds[model])
                        pred = normalize(raw, predicted)
                        original = base.score(row, predicted)
                        exact[model].update(original)
                        for cutoff in CUTOFFS:
                            credit, matched = optimal_credit(pred, gold, cutoff)
                            counts = aggregate[model][cutoff]
                            counts["credit"] += credit
                            counts["matched"] += matched
                            counts["n_pred"] += len(pred)
                            counts["n_gold"] += len(gold)
            for model in MODELS:
                reference = metrics["test_sets"][corpus][model]
                _, _, exact_f1 = base.prf(exact[model]["span_tp"], exact[model]["span_pred"],
                                          exact[model]["span_ref"])
                if (reference["n_documents"] != len(test)
                        or any(exact[model][key] != reference[key]
                               for key in ("span_tp", "span_pred", "span_ref"))
                        or abs(exact_f1 - reference["span_f1"]) > 1e-12):
                    raise ValueError(f"Existing exact result not reproduced: {corpus}/{model}")
                for cutoff in CUTOFFS:
                    counts = aggregate[model][cutoff]
                    precision, recall, f1 = metric(counts["credit"],
                                                   int(counts["n_pred"]), int(counts["n_gold"]))
                    rows_out.append({"corpus": corpus, "model": model, "n_test": len(test),
                                     "checkpoint": selection["M2_candidate"] if model == "M2" else "base",
                                     "threshold": thresholds[model], "iou_cutoff": cutoff,
                                     "n_pred": int(counts["n_pred"]), "n_gold": int(counts["n_gold"]),
                                     "n_matched": int(counts["matched"]), "iou_credit": counts["credit"],
                                     "soft_precision": precision, "soft_recall": recall,
                                     "soft_f1": f1, "exact_f1": exact_f1})
            audit[corpus] = {"test_ids_sha256": base.digest(bundle / "folds" / f"target_{corpus}" / "split_ids.json"),
                             "selection_sha256": base.digest(selection_path),
                             "metrics_sha256": base.digest(metrics_path),
                             "adapted_cache_sha256": base.digest(source_run / "heldout_probabilities.sqlite3")}
            print(f"{corpus}: {len(test)} test narratives checked", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    columns = tuple(rows_out[0])
    base.write_csv(out / "soft_span_details.csv", rows_out, columns)
    main = [row for row in rows_out if row["iou_cutoff"] == 0.75]
    by_corpus = collections.defaultdict(dict)
    for row in main:
        by_corpus[row["corpus"]][row["model"]] = row
    comparison = []
    for corpus in base.CORPORA:
        scores = by_corpus[corpus]
        comparison.append({"corpus": corpus, "n_test": scores["M0"]["n_test"],
                           **{f"{model}_exact_f1": scores[model]["exact_f1"] for model in MODELS},
                           **{f"{model}_soft_f1": scores[model]["soft_f1"] for model in MODELS},
                           "gain_M2_minus_M1": scores["M2"]["soft_f1"] - scores["M1"]["soft_f1"]})
    macro = {"corpus": "MACRO_6_CORPUS", "n_test": sum(r["n_test"] for r in comparison)}
    for key in comparison[0]:
        if key not in {"corpus", "n_test"}:
            macro[key] = sum(row[key] for row in comparison) / len(comparison)
    comparison.append(macro)
    base.write_csv(out / "soft_span_comparison.csv", comparison, tuple(comparison[0]))
    svg_chart(out / "soft_span_comparison.svg", rows_out)
    report = ["# Soft Span F1 — expérience in-domain SaT", "",
              "Scores descriptifs sur les mêmes 15 % de récits tests réservés que le rapport initial.",
              "M0 et M1 utilisent SaT initial ; M2 utilise l'adaptateur et le checkpoint",
              "retenus par F1 exact sur la validation du corpus source. Aucun choix n'a été",
              "refait à partir des résultats Soft Span.", "",
              "Pour chaque récit, les segments sont appariés un-à-un afin de maximiser la somme",
              "des IoU en caractères, après retrait des espaces périphériques. Une paire",
              "reçoit son IoU si celui-ci atteint 0,75, sinon zéro. La précision et le rappel",
              "divisent la somme de ces crédits par les nombres de segments prédits et gold ;",
              "leur F1 est calculé en micro par corpus. La ligne macro moyenne les six corpus.",
              "Les segments prédits sans correspondant, y compris dans un passage exclu",
              "du gold, restent au dénominateur.", "",
              "| Corpus | Récits tests | M0 exact | M1 exact | M2 exact | M0 soft | M1 soft | M2 soft | Gain M2−M1 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in comparison:
        report.append("| {corpus} | {n_test} | {M0_exact_f1:.3f} | {M1_exact_f1:.3f} | "
                      "{M2_exact_f1:.3f} | {M0_soft_f1:.3f} | {M1_soft_f1:.3f} | "
                      "{M2_soft_f1:.3f} | {gain_M2_minus_M1:+.3f} |".format(**row))
    report.extend(["", "`soft_span_details.csv` contient aussi précision, rappel et les seuils",
                   "d'IoU 0,50 et 0,90. `soft_span_comparison.svg` est le graphique.", "",
                   "Cette métrique a été introduite après examen des résultats exacts ; elle",
                   "constitue ici une analyse complémentaire, non une nouvelle sélection de modèle.",
                   "Un chevauchement élevé ne garantit pas que le sens ou le rôle du fait soit préservé.", ""])
    (out / "README.md").write_text("\n".join(report), encoding="utf-8")
    base.write_json(out / "manifest.json", {
        "reference": base.REFERENCE, "design": "post-hoc descriptive, fixed 15% in-domain test",
        "selection": "original exact-span validation selection; no reselection for soft metric",
        "method": "Python character IoU after peripheral whitespace trimming; optimal one-to-one matching within each narrative; credit=IoU if IoU>=cutoff; micro per corpus, macro across six corpora",
        "cutoffs": CUTOFFS, "primary_descriptive_cutoff": 0.75,
        "unmatched_predictions": "included in denominator, including nonfactual text",
        "bundle_manifest_sha256": manifest_hash, "base_cache_sha256": base.digest(bundle / "base_probabilities.sqlite3"),
        "source_files": audit,
        "outputs_sha256": {name: base.digest(out / name) for name in
                           ("soft_span_details.csv", "soft_span_comparison.csv",
                            "soft_span_comparison.svg", "README.md")},
    })
    return comparison


def calculate_transfer(bundle: Path, run_root: Path, out: Path) -> tuple[list[dict], dict]:
    """Recompute all six-by-six held-out comparisons from the saved caches."""
    manifest_hash = base.digest(bundle / "manifest.json")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest["reference"] != base.REFERENCE or manifest["fold_order"] != list(base.CORPORA):
        raise ValueError("Unexpected in-domain bundle")
    tests = {}
    for target in base.CORPORA:
        folder = bundle / "folds" / f"target_{target}"
        rows = base.read_jsonl(folder / "test.jsonl")
        if (len(rows) != manifest["report"][target]["split"]["test"]
                or len({row["document_id"] for row in rows}) != len(rows)):
            raise ValueError(f"Invalid held-out test partition: {target}")
        for row in rows:
            base.validate_gold(row)
            if row["corpus"] != target:
                raise ValueError(f"Test corpus mismatch: {target}")
        tests[target] = rows
    detailed = []
    source_audit = {}
    with closing(base._probability_connection(bundle / "base_probabilities.sqlite3")) as baseline:
        for source in base.CORPORA:
            run = run_root / f"source_{source}"
            selection_path = run / "selection.json"
            metrics_path = run / "metrics.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            binding = json.loads((run / "heldout_probabilities.binding.json").read_text(encoding="utf-8"))
            if (selection["source"] != source or metrics["source"] != source
                    or selection["bundle_manifest_sha256"] != manifest_hash
                    or metrics["bundle_manifest_sha256"] != manifest_hash
                    or metrics["selection_sha256"] != base.digest(selection_path)
                    or binding["bundle_manifest_sha256"] != manifest_hash
                    or binding["adapter_sha256"] != metrics["adapter_sha256"]
                    or binding["head_sha256"] != metrics["head_sha256"]):
                raise ValueError(f"Selection/model/data provenance mismatch: {source}")
            thresholds = {"M0": base.DEFAULT, "M1": selection["M1_threshold"],
                          "M2": selection["M2_threshold"]}
            with closing(base._probability_connection(run / "heldout_probabilities.sqlite3")) as adapted:
                for target in base.CORPORA:
                    totals = {model: {cutoff: collections.defaultdict(float) for cutoff in CUTOFFS}
                              for model in MODELS}
                    exact = {model: collections.Counter() for model in MODELS}
                    for row in tests[target]:
                        raw = row["raw_text"]
                        gold = normalize(raw, [(unit["start"], unit["end"]) for unit in row["units"]])
                        base_probs = base.get_probs(baseline, row)
                        adapted_probs = base.get_probs(adapted, row)
                        for model in MODELS:
                            probs = adapted_probs if model == "M2" else base_probs
                            predicted = cached_spans(raw, probs, thresholds[model])
                            pred = normalize(raw, predicted)
                            exact[model].update(base.score(row, predicted))
                            for cutoff in CUTOFFS:
                                credit, matched = optimal_credit(pred, gold, cutoff)
                                counts = totals[model][cutoff]
                                counts["credit"] += credit
                                counts["matched"] += matched
                                counts["n_pred"] += len(pred)
                                counts["n_gold"] += len(gold)
                    for model in MODELS:
                        reference = metrics["test_sets"][target][model]
                        _, _, exact_f1 = base.prf(exact[model]["span_tp"],
                                                  exact[model]["span_pred"], exact[model]["span_ref"])
                        if (reference["n_documents"] != len(tests[target])
                                or any(exact[model][key] != reference[key]
                                       for key in ("span_tp", "span_pred", "span_ref"))
                                or abs(exact_f1 - reference["span_f1"]) > 1e-12):
                            raise ValueError(f"Existing exact result not reproduced: {source}/{target}/{model}")
                        for cutoff in CUTOFFS:
                            counts = totals[model][cutoff]
                            precision, recall, f1 = metric(counts["credit"],
                                                           int(counts["n_pred"]),
                                                           int(counts["n_gold"]))
                            detailed.append({"source": source, "target": target,
                                             "comparison": "in_domain" if source == target else "cross_domain",
                                             "model": model, "n_test": len(tests[target]),
                                             "checkpoint": selection["M2_candidate"] if model == "M2" else "base",
                                             "threshold": thresholds[model], "iou_cutoff": cutoff,
                                             "n_pred": int(counts["n_pred"]),
                                             "n_gold": int(counts["n_gold"]),
                                             "n_matched": int(counts["matched"]),
                                             "iou_credit": counts["credit"],
                                             "soft_precision": precision, "soft_recall": recall,
                                             "soft_f1": f1, "exact_f1": exact_f1})
                    print(f"{source} -> {target}: {len(tests[target])} tests checked", flush=True)
            source_audit[source] = {"selection_sha256": base.digest(selection_path),
                                    "metrics_sha256": base.digest(metrics_path),
                                    "adapted_cache_sha256": base.digest(run / "heldout_probabilities.sqlite3")}
    # The recalculated diagonal must also agree with the previous post-hoc report.
    with (out / "soft_span_details.csv").open(encoding="utf-8-sig", newline="") as stream:
        prior = {(row["corpus"], row["model"], float(row["iou_cutoff"])): row
                 for row in csv.DictReader(stream)}
    for row in detailed:
        if row["source"] == row["target"]:
            previous = prior[row["source"], row["model"], row["iou_cutoff"]]
            if (abs(row["soft_f1"] - float(previous["soft_f1"])) > 1e-12
                    or row["n_test"] != int(previous["n_test"])):
                raise ValueError("Transfer diagonal disagrees with in-domain report")
    base.write_csv(out / "soft_span_transfer_details.csv", detailed, tuple(detailed[0]))
    core = {(row["source"], row["target"], row["model"]): row
            for row in detailed if row["iou_cutoff"] == 0.75}
    for model in MODELS:
        matrix = [{"source": source,
                   **{target: core[source, target, model]["soft_f1"]
                      for target in base.CORPORA}} for source in base.CORPORA]
        base.write_csv(out / f"soft_span_transfer_{model}.csv", matrix, tuple(matrix[0]))
    gains = [{"source": source,
              **{target: core[source, target, "M2"]["soft_f1"]
                 - core[source, target, "M1"]["soft_f1"] for target in base.CORPORA}}
             for source in base.CORPORA]
    base.write_csv(out / "soft_span_transfer_gain_M2_minus_M1.csv", gains, tuple(gains[0]))
    transfer_svg(out / "soft_span_transfer.svg", detailed)
    summary = {}
    for model in MODELS:
        values = [core[source, target, model]["soft_f1"]
                  for source in base.CORPORA for target in base.CORPORA]
        diagonal = [core[corpus, corpus, model]["soft_f1"] for corpus in base.CORPORA]
        cross = [core[source, target, model]["soft_f1"]
                 for source in base.CORPORA for target in base.CORPORA if source != target]
        summary[model] = {"mean_in_domain": sum(diagonal) / len(diagonal),
                          "mean_cross_domain": sum(cross) / len(cross),
                          "mean_all_cells": sum(values) / len(values)}
    summary["gain_M2_minus_M1"] = {
        "mean_in_domain": summary["M2"]["mean_in_domain"] - summary["M1"]["mean_in_domain"],
        "mean_cross_domain": summary["M2"]["mean_cross_domain"] - summary["M1"]["mean_cross_domain"]}
    base.write_json(out / "soft_span_transfer_summary.json", summary)
    positive_cross = sum(gains[i][target] > 0 for i, source in enumerate(base.CORPORA)
                         for target in base.CORPORA if source != target)
    matrix_lines = ["", "## Matrice de transfert : M2", "",
                    "Chaque ligne est le seul corpus utilisé pour adapter SaT ; chaque colonne",
                    "est le corpus testé sur ses mêmes 15 % de récits réservés. Les cases de",
                    "la diagonale sont in-domain. M1 et M2 ont chacun conservé le seuil choisi",
                    "sur la validation du corpus source, selon le F1 exact.", "",
                    "| Source / Test | BTP | Métal. | Chimie | Bois | Service temp. | Nicollin |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for source in base.CORPORA:
        values = [f"{core[source, target, 'M2']['soft_f1']:.3f}" for target in base.CORPORA]
        matrix_lines.append("| " + source + " | " + " | ".join(values) + " |")
    matrix_lines.extend(["", "Moyenne des six cases in-domain : "
                         f"{summary['M2']['mean_in_domain']:.3f} ; moyenne des 30 cases hors domaine : "
                         f"{summary['M2']['mean_cross_domain']:.3f}.",
                         "Gain moyen M2−M1 hors domaine : "
                         f"{summary['gain_M2_minus_M1']['mean_cross_domain']:+.3f} ; "
                         f"gain positif dans {positive_cross}/30 cases.",
                         "Les résultats de transfert d'un seul corpus source ne remplacent pas",
                         "l'expérience principale d'entraînement sur cinq corpus LODO.", "",
                         "`soft_span_transfer.svg` montre M2 et son gain sur M1. Les CSV",
                         "`soft_span_transfer_M0.csv`, `soft_span_transfer_M1.csv`,",
                         "`soft_span_transfer_M2.csv` et `soft_span_transfer_gain_M2_minus_M1.csv`",
                         "contiennent les quatre matrices à IoU ≥ 0,75 ; le fichier détaillé",
                         "contient aussi précision, rappel et les seuils 0,50 et 0,90.", ""])
    readme_path = out / "README.md"
    with readme_path.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(matrix_lines))
    manifest_path = out / "manifest.json"
    report_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report_manifest["design"] = "post-hoc descriptive, fixed 15% test per target; 6x6 single-source transfer"
    report_manifest["transfer"] = {
        "test_rule": "same held-out 15% narratives for every source model of a target corpus",
        "selection_rule": "checkpoint and thresholds selected only on source validation by exact-span F1",
        "cross_domain_cells": 30, "in_domain_cells": 6,
        "source_files": source_audit,
    }
    names = ["soft_span_transfer_details.csv", "soft_span_transfer.svg",
             "soft_span_transfer_gain_M2_minus_M1.csv", "soft_span_transfer_summary.json",
             *(f"soft_span_transfer_{model}.csv" for model in MODELS)]
    report_manifest["outputs_sha256"].update({name: base.digest(out / name) for name in names})
    report_manifest["outputs_sha256"]["README.md"] = base.digest(readme_path)
    base.write_json(manifest_path, report_manifest)
    return detailed, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.run_root / "soft_span_iou_v1"
    results = calculate(args.bundle.resolve(), args.run_root.resolve(), out.resolve())
    _, transfer_summary = calculate_transfer(args.bundle.resolve(), args.run_root.resolve(), out.resolve())
    for row in results:
        print(f"{row['corpus']}: M0={row['M0_soft_f1']:.4f}, M1={row['M1_soft_f1']:.4f}, "
              f"M2={row['M2_soft_f1']:.4f}, gain={row['gain_M2_minus_M1']:+.4f}")
    print(f"Results: {out.resolve()}")
    print("Cross-domain means:", {model: round(transfer_summary[model]["mean_cross_domain"], 4)
                                  for model in MODELS})


if __name__ == "__main__":
    main()
