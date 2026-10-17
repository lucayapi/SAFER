#!/usr/bin/env python3
"""Recompute publication tables from the private SaT LODO prediction caches.

The reference is AI direct preannotation, not independently validated human gold.
No model inference or retraining is performed. Source-only threshold selection is
redone with whitespace-neutralized exact-span F1, leaving target tests untouched.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

import sat_lodo as experiment


ROOT = Path(__file__).resolve().parents[2]
ROLES = ("A0", "A1", "B", "C")
MODELS = ("M0", "M1", "M2")
LABELS = {
    "M0": "SaT initial, seuil par défaut",
    "M1": "SaT initial, seuil calibré sur les sources",
    "M2": "SaT + LoRA, seuil calibré sur les sources",
}
DOMAIN_NAMES = {
    "btp": "BTP", "metallurgie": "Métallurgie",
    "caou_chimie_plas": "Chimie–plastiques",
    "bois_textille": "Bois–textile",
    "service_temporaire": "Service temporaire", "nicollin": "Nicollin",
    "macro_6_domains": "Moyenne macro (six corpus)",
}
COUNT_KEYS = (
    "span_tp", "span_pred", "span_ref", "boundary_tp", "boundary_pred",
    "boundary_ref", "fusion", "pred_segments", "supported_pred_segments",
    "fragmented", "gold_units",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def normalized_spans(text: str, spans: list[tuple[int, int]]) -> set[tuple[int, int]]:
    return {pair for start, end in spans
            if (pair := trim_span(text, start, end))[0] < pair[1]}


def nonspace_positions(text: str) -> list[int]:
    return [index for index, char in enumerate(text) if not char.isspace()]


def nonspace_rank(positions: list[int], offset: int) -> int:
    return bisect.bisect_right(positions, offset - 1)


def boundary_set(positions: list[int], spans: list[tuple[int, int]]) -> set[int]:
    total = len(positions)
    return {rank for start, end in spans for offset in (start, end)
            if 0 < (rank := nonspace_rank(positions, offset)) < total}


def prf(tp: int, pred: int, ref: int) -> tuple[float, float, float]:
    precision = tp / pred if pred else 0.0
    recall = tp / ref if ref else 0.0
    f1 = 2 * tp / (pred + ref) if pred + ref else 0.0
    return precision, recall, f1


def score_span(text: str, gold: list[tuple[int, int]],
               predicted: list[tuple[int, int]]) -> tuple[int, int, int]:
    reference = normalized_spans(text, gold)
    prediction = normalized_spans(text, predicted)
    return len(reference & prediction), len(prediction), len(reference)


def score_document(row: dict, predicted: list[tuple[int, int]]) -> dict[str, int]:
    """One document's sufficient statistics; whitespace alone never overlaps."""
    text = row["raw_text"]
    gold_raw = [(unit["start"], unit["end"]) for unit in row["units"]]
    for start, end in predicted:
        if not 0 <= start < end <= len(text):
            raise ValueError(f"Invalid prediction offset: {row['document_id']}")
    if any(right[0] < left[1] for left, right in zip(predicted, predicted[1:])):
        raise ValueError(f"Overlapping predictions: {row['document_id']}")
    gold = normalized_spans(text, gold_raw)
    pred = normalized_spans(text, predicted)
    positions = nonspace_positions(text)
    gold_cuts = boundary_set(positions, gold_raw)
    pred_cuts = boundary_set(positions, predicted)
    counts = collections.Counter({
        "span_tp": len(gold & pred), "span_pred": len(pred), "span_ref": len(gold),
        "boundary_tp": len(gold_cuts & pred_cuts),
        "boundary_pred": len(pred_cuts), "boundary_ref": len(gold_cuts),
        "pred_segments": len(pred), "gold_units": len(gold_raw),
    })
    for unit in row["units"]:
        role = unit["role"]
        if role in ROLES:
            counts[f"role_{role}_ref"] += 1
            counts[f"role_{role}_tp"] += trim_span(
                text, unit["start"], unit["end"]) in pred

    # A fusion is one predicted segment containing at least two *whole* facts.
    for start, end in pred:
        contained = sum(start <= a and b <= end for a, b in gold)
        counts["fusion"] += contained >= 2
        counts["supported_pred_segments"] += contained >= 1

    # A fragmented fact meets at least two predicted segments on nonspace chars.
    pred_nonspace = [(nonspace_rank(positions, a), nonspace_rank(positions, b))
                     for a, b in pred]
    for start, end in gold:
        a, b = nonspace_rank(positions, start), nonspace_rank(positions, end)
        overlaps = sum(max(a, x) < min(b, y) for x, y in pred_nonspace)
        counts["fragmented"] += overlaps >= 2
    return dict(counts)


def load_predictions(db: sqlite3.Connection, model: str, row: dict) -> dict[str, list[tuple[int, int]]]:
    found = db.execute(
        "SELECT spans_json FROM predictions WHERE model=? AND corpus=? AND document_id=?",
        (model, row["corpus"], row["document_id"]),
    ).fetchone()
    if not found:
        raise ValueError(f"Missing {model} predictions for {row['corpus']}/{row['document_id']}")
    result = json.loads(found[0])
    if set(result) != {f"{threshold:.2f}" for threshold in experiment.THRESHOLDS}:
        raise ValueError(f"Incomplete threshold cache: {row['document_id']}")
    return {key: [tuple(pair) for pair in pairs] for key, pairs in result.items()}


def select_thresholds(validation: list[dict], db: sqlite3.Connection,
                      target: str) -> tuple[dict[str, float], list[dict]]:
    counts = collections.defaultdict(collections.Counter)
    for row in validation:
        gold = [(unit["start"], unit["end"]) for unit in row["units"]]
        for model in ("base", "lora"):
            cached = load_predictions(db, model, row)
            for threshold in experiment.THRESHOLDS:
                tp, pred, ref = score_span(
                    row["raw_text"], gold, cached[f"{threshold:.2f}"])
                counts[model, threshold, row["corpus"]].update(
                    {"tp": tp, "pred": pred, "ref": ref})

    domains = [domain for domain in experiment.CORPORA if domain != target]
    curves = []
    selected = {}
    for model in ("base", "lora"):
        candidates = []
        for threshold in experiment.THRESHOLDS:
            per_domain = {}
            for domain in domains:
                c = counts[model, threshold, domain]
                if not c["ref"]:
                    raise ValueError(f"Empty source validation: {target}/{domain}")
                per_domain[domain] = prf(c["tp"], c["pred"], c["ref"])[2]
            macro_f1 = sum(per_domain.values()) / len(domains)
            candidates.append((threshold, macro_f1, per_domain))
        winner = min(candidates, key=lambda item: (
            -item[1], abs(item[0] - experiment.DEFAULT), item[0]))
        selected[model] = winner[0]
        for threshold, macro_f1, per_domain in candidates:
            curves.append({"target": target, "model": model,
                           "threshold": threshold,
                           "source_macro_span_f1_trimmed": macro_f1,
                           "selected": threshold == winner[0],
                           **{f"f1_{domain}": per_domain.get(domain)
                              for domain in experiment.CORPORA}})
    return selected, curves


def total_counts(rows: list[dict]) -> collections.Counter:
    total = collections.Counter()
    for row in rows:
        total.update(row)
    return total


def metric_row(target: str, model: str, threshold: float,
               n_documents: int, counts: collections.Counter) -> dict:
    sp, sr, sf = prf(counts["span_tp"], counts["span_pred"], counts["span_ref"])
    bp, br, bf = prf(counts["boundary_tp"], counts["boundary_pred"],
                     counts["boundary_ref"])
    return {"target": target, "model": model, "model_description": LABELS[model],
            "threshold": threshold, "n_documents": n_documents,
            "span_tp": counts["span_tp"], "span_pred": counts["span_pred"],
            "span_ref": counts["span_ref"], "span_precision": sp,
            "span_recall": sr, "span_f1": sf,
            "boundary_tp": counts["boundary_tp"],
            "boundary_pred": counts["boundary_pred"],
            "boundary_ref": counts["boundary_ref"],
            "boundary_precision": bp, "boundary_recall": br,
            "boundary_f1": bf,
            "fusion_count": counts["fusion"],
            "pred_segments": counts["pred_segments"],
            "supported_pred_segments": counts["supported_pred_segments"],
            "fusion_rate_all_predicted": counts["fusion"] / counts["pred_segments"]
            if counts["pred_segments"] else 0.0,
            "fusion_rate_supported": counts["fusion"] / counts["supported_pred_segments"]
            if counts["supported_pred_segments"] else 0.0,
            "fragmented_count": counts["fragmented"],
            "gold_units": counts["gold_units"],
            "fragmentation_rate": counts["fragmented"] / counts["gold_units"]
            if counts["gold_units"] else 0.0}


def bootstrap_gain(left: list[dict], right: list[dict], seed: int,
                   n_bootstrap: int) -> tuple[np.ndarray, np.ndarray]:
    """Paired document bootstrap; fitted models and chosen thresholds stay fixed."""
    if len(left) != len(right) or not left:
        raise ValueError("Bootstrap requires paired, nonempty document statistics")
    names = ("span_tp", "span_pred", "span_ref", "boundary_tp",
             "boundary_pred", "boundary_ref")
    first = np.asarray([[row.get(key, 0) for key in names] for row in left],
                       dtype=np.int64)
    second = np.asarray([[row.get(key, 0) for key in names] for row in right],
                        dtype=np.int64)
    rng = np.random.default_rng(seed)
    span_deltas = np.empty(n_bootstrap)
    boundary_deltas = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        sample = rng.integers(len(left), size=len(left))
        a, b = first[sample].sum(axis=0), second[sample].sum(axis=0)
        span_deltas[index] = 2 * b[0] / (b[1] + b[2]) - 2 * a[0] / (a[1] + a[2])
        boundary_deltas[index] = 2 * b[3] / (b[4] + b[5]) - 2 * a[3] / (a[4] + a[5])
    return span_deltas, boundary_deltas


def draw_figures(out: Path, metrics: list[dict], gains: list[dict],
                 roles: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = list(experiment.CORPORA)
    chosen = {row["target"]: row for row in gains if row["target"] != "macro_6_domains"}
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    xs = np.arange(len(targets))
    values = np.array([chosen[t]["span_f1_gain"] for t in targets])
    low = np.array([chosen[t]["span_ci_low"] for t in targets])
    high = np.array([chosen[t]["span_ci_high"] for t in targets])
    ax.errorbar(xs, values, yerr=[values - low, high - values], fmt="o",
                color="#24577d", capsize=4, linewidth=1.4)
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_xticks(xs, [DOMAIN_NAMES[t] for t in targets], rotation=25, ha="right")
    ax.set_ylabel("Gain M2 − M1 : F1 des segments (espaces neutralisés)")
    ax.set_title("Transfert LoRA vers un corpus absent de l'entraînement")
    ax.grid(axis="y", alpha=0.25)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"figure_1_gains_span_f1.{suffix}", dpi=220)
    plt.close(fig)

    by_key = {(row["target"], row["model"], row["role"]): row["recall"]
              for row in roles if row["target"] != "macro_6_domains"}
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), sharey=True)
    fig.subplots_adjust(left=.07, right=.99, top=.87, bottom=.10,
                        hspace=.24, wspace=.04)
    colors = {"M0": "#788897", "M1": "#376b95", "M2": "#b36a44"}
    for target, ax in zip(targets, axes.flat):
        x = np.arange(len(ROLES))
        for j, model in enumerate(MODELS):
            y = [by_key[target, model, role] for role in ROLES]
            ax.bar(x + (j - 1) * 0.24, y, 0.23, label=model,
                   color=colors[model])
        ax.set_xticks(x, ROLES)
        ax.set_title(DOMAIN_NAMES[target])
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.18)
    axes[0, 0].set_ylabel("Rappel exact (espaces neutralisés)")
    axes[1, 0].set_ylabel("Rappel exact (espaces neutralisés)")
    fig.legend(MODELS, loc="upper center", ncol=3, bbox_to_anchor=(0.5, .98))
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"figure_2_rappel_par_role.{suffix}", dpi=220)
    plt.close(fig)


def generate(bundle: Path, runs: Path, out: Path, n_bootstrap: int, seed: int) -> None:
    if n_bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    if out.exists():
        raise FileExistsError(f"Results directory already exists: {out}")
    manifest = experiment.verify_bundle(bundle)
    if manifest.get("data_policy") != experiment.DATA_POLICY:
        raise ValueError("Expected the v3 direct-except-quality-control bundle")
    manifest_sha = sha256(bundle / "manifest.json")
    out.mkdir(parents=True)
    curve_rows, metrics, role_rows, error_rows, gain_rows = [], [], [], [], []
    threshold_rows, cache_hashes = [], {}
    bootstraps = {}
    for target_index, target in enumerate(experiment.CORPORA):
        folder = bundle / "folds" / f"target_{target}"
        run = runs / f"target_{target}"
        recorded = json.loads((run / "bundle_binding.json").read_text(encoding="utf-8"))
        completed = json.loads((run / "training_done.json").read_text(encoding="utf-8"))
        if recorded["bundle_manifest_sha256"] != manifest_sha or \
           completed["bundle_manifest_sha256"] != manifest_sha or \
           completed["target"] != target:
            raise ValueError(f"Run and bundle mismatch: {target}")
        validation, test = read_jsonl(folder / "val.jsonl"), read_jsonl(folder / "test.jsonl")
        cache = run / "predictions.sqlite3"
        cache_hashes[target] = sha256(cache)
        db = sqlite3.connect(f"file:{cache.resolve().as_posix()}?mode=ro", uri=True)
        try:
            expected = len(validation) + len(test)
            sizes = dict(db.execute("SELECT model, COUNT(*) FROM predictions GROUP BY model"))
            if sizes != {"base": expected, "lora": expected}:
                raise ValueError(f"Incomplete prediction cache: {target}: {sizes}")
            selected, curves = select_thresholds(validation, db, target)
            curve_rows.extend(curves)
            print(f"{target}: source-selected thresholds base={selected['base']:.2f}, "
                  f"LoRA={selected['lora']:.2f}", flush=True)
            threshold_rows.append({"target": target,
                                   "selection_metric": "source macro of per-domain micro trimmed-span F1",
                                   "M0": experiment.DEFAULT, "M1": selected["base"],
                                   "M2": selected["lora"],
                                   "n_source_validation": len(validation),
                                   "n_target_test": len(test)})
            doc_counts = {model: [] for model in MODELS}
            for row in test:
                base = load_predictions(db, "base", row)
                lora = load_predictions(db, "lora", row)
                predictions = {
                    "M0": base[f"{experiment.DEFAULT:.2f}"],
                    "M1": base[f"{selected['base']:.2f}"],
                    "M2": lora[f"{selected['lora']:.2f}"],
                }
                for model, spans in predictions.items():
                    doc_counts[model].append(score_document(row, spans))
            for model in MODELS:
                c = total_counts(doc_counts[model])
                threshold = experiment.DEFAULT if model == "M0" else selected[
                    "base" if model == "M1" else "lora"]
                result = metric_row(target, model, threshold, len(test), c)
                metrics.append(result)
                error_rows.append({key: result[key] for key in (
                    "target", "model", "n_documents", "fusion_count", "pred_segments",
                    "supported_pred_segments", "fusion_rate_all_predicted",
                    "fusion_rate_supported", "fragmented_count", "gold_units",
                    "fragmentation_rate")})
                for role in ROLES:
                    reference = c[f"role_{role}_ref"]
                    role_rows.append({"target": target, "model": model, "role": role,
                                      "matched": c[f"role_{role}_tp"],
                                      "reference": reference,
                                      "recall": c[f"role_{role}_tp"] / reference
                                      if reference else None})
            left = next(item for item in metrics if item["target"] == target and item["model"] == "M1")
            right = next(item for item in metrics if item["target"] == target and item["model"] == "M2")
            span_boot, boundary_boot = bootstrap_gain(
                doc_counts["M1"], doc_counts["M2"], seed + target_index,
                n_bootstrap)
            bootstraps[target] = (span_boot, boundary_boot)
            gain_rows.append({"target": target, "n_documents": len(test),
                              "span_f1_gain": right["span_f1"] - left["span_f1"],
                              "span_ci_low": float(np.quantile(span_boot, .025)),
                              "span_ci_high": float(np.quantile(span_boot, .975)),
                              "boundary_f1_gain": right["boundary_f1"] - left["boundary_f1"],
                              "boundary_ci_low": float(np.quantile(boundary_boot, .025)),
                              "boundary_ci_high": float(np.quantile(boundary_boot, .975))})
        finally:
            db.close()

    # Macro-domain rows average the six within-domain scores, not pooled counts.
    for model in MODELS:
        subset = [row for row in metrics if row["model"] == model]
        metrics.append({"target": "macro_6_domains", "model": model,
                        "model_description": LABELS[model], "threshold": None,
                        "n_documents": sum(row["n_documents"] for row in subset),
                        **{key: sum(row[key] for row in subset) / len(subset)
                           for key in ("span_precision", "span_recall", "span_f1",
                                       "boundary_precision", "boundary_recall",
                                       "boundary_f1")}})
        for role in ROLES:
            scores = [row["recall"] for row in role_rows
                      if row["model"] == model and row["role"] == role]
            role_rows.append({"target": "macro_6_domains", "model": model,
                              "role": role, "matched": None, "reference": None,
                              "recall": sum(scores) / len(scores)})
    macro_span = np.mean([bootstraps[target][0] for target in experiment.CORPORA], axis=0)
    macro_boundary = np.mean([bootstraps[target][1] for target in experiment.CORPORA], axis=0)
    macro_by_model = {row["model"]: row for row in metrics if row["target"] == "macro_6_domains"}
    gain_rows.append({"target": "macro_6_domains",
                      "n_documents": sum(row["n_documents"] for row in gain_rows),
                      "span_f1_gain": macro_by_model["M2"]["span_f1"] - macro_by_model["M1"]["span_f1"],
                      "span_ci_low": float(np.quantile(macro_span, .025)),
                      "span_ci_high": float(np.quantile(macro_span, .975)),
                      "boundary_f1_gain": macro_by_model["M2"]["boundary_f1"] - macro_by_model["M1"]["boundary_f1"],
                      "boundary_ci_low": float(np.quantile(macro_boundary, .025)),
                      "boundary_ci_high": float(np.quantile(macro_boundary, .975))})

    for name, fields in (
        ("tableau_1_segments.csv", ["target", "model", "threshold", "n_documents",
                                    "span_tp", "span_pred", "span_ref", "span_precision",
                                    "span_recall", "span_f1"]),
        ("tableau_2_frontieres.csv", ["target", "model", "threshold",
                                         "n_documents", "boundary_tp", "boundary_pred",
                                         "boundary_ref", "boundary_precision",
                                         "boundary_recall", "boundary_f1"]),
    ):
        write_csv(out / name, metrics, fields)
    write_csv(out / "tableau_3_rappel_par_role.csv", role_rows,
              ["target", "model", "role", "matched", "reference", "recall"])
    write_csv(out / "tableau_4_fusions_fragmentations.csv", error_rows,
              ["target", "model", "n_documents", "fusion_count", "pred_segments",
               "supported_pred_segments", "fusion_rate_all_predicted",
               "fusion_rate_supported", "fragmented_count", "gold_units",
               "fragmentation_rate"])
    write_csv(out / "gains_ic95.csv", gain_rows, list(gain_rows[0]))
    write_csv(out / "seuils_retenus.csv", threshold_rows, list(threshold_rows[0]))
    write_csv(out / "courbes_validation_sources.csv", curve_rows, list(curve_rows[0]))
    by_target_model = {(row["target"], row["model"]): row for row in metrics}
    by_target_gain = {row["target"]: row for row in gain_rows}
    main_rows = []
    for target in (*experiment.CORPORA, "macro_6_domains"):
        m0, m1, m2 = (by_target_model[target, model] for model in MODELS)
        gain = by_target_gain[target]
        main_rows.append({"target": target, "corpus_label": DOMAIN_NAMES[target],
                          "n_documents": m0["n_documents"],
                          "M0_span_f1": m0["span_f1"], "M1_span_f1": m1["span_f1"],
                          "M2_span_f1": m2["span_f1"],
                          "gain_M2_minus_M1": gain["span_f1_gain"],
                          "gain_ci95_low": gain["span_ci_low"],
                          "gain_ci95_high": gain["span_ci_high"]})
    write_csv(out / "tableau_principal.csv", main_rows, list(main_rows[0]))
    draw_figures(out, metrics, gain_rows, role_rows)

    positive = sum(row["span_f1_gain"] > 0 for row in gain_rows
                   if row["target"] != "macro_6_domains")
    result = {"reference": "AI_DIRECT_PREANNOTATIONS_NOT_HUMAN_GOLD",
              "quality_control_excluded_by_bundle": True,
              "bundle_manifest_sha256": manifest_sha,
              "prediction_cache_sha256": cache_hashes,
              "source_threshold_selection": "macro of five source-domain micro trimmed-span F1",
              "bootstrap": {"replicates": n_bootstrap, "seed": seed,
                            "unit": "narrative, paired M1/M2, target threshold fixed",
                            "method": "percentile 2.5/97.5; independent resampling per domain"},
              "n_positive_exact_span_domains": positive,
              "macro_gain_span_f1": gain_rows[-1]["span_f1_gain"]}
    (out / "provenance.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table_lines = ["| Corpus cible | Récits | M0 | M1 | M2 | Gain M2 − M1 [IC 95 %] |",
                   "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in main_rows:
        table_lines.append(
            f"| {row['corpus_label']} | {row['n_documents']} | {row['M0_span_f1']:.3f} | "
            f"{row['M1_span_f1']:.3f} | {row['M2_span_f1']:.3f} | "
            f"{row['gain_M2_minus_M1']:+.3f} "
            f"[{row['gain_ci95_low']:+.3f} ; {row['gain_ci95_high']:+.3f}] |")
    main_table = "\n".join(table_lines)
    notes = f"""# Résultats exploratoires SaT LODO — six corpus

Référence : **préannotations IA directes, non validées humainement**. Le bundle
`sat12l_sm_lodo_v3` exclut les récits des deux fichiers de contrôle qualité ;
le script vérifie son empreinte avant tout calcul. Les unités `uncertain=true`
ou les récits comportant d'autres `issues` restent présents conformément à la
politique v3. Ces scores ne sont pas des performances sur un gold humain.

## Tableau principal : F1 des segments, espaces périphériques neutralisés

{main_table}

La ligne macro est la moyenne arithmétique des six corpus. Son intervalle de
confiance porte sur la moyenne des gains appariés et ne mesure pas l'incertitude
liée aux annotations, aux graines d'entraînement ou au checkpoint final.

Chaque rotation entraîne sur cinq corpus, utilise leurs 20 % de validation pour
sélectionner séparément les seuils M1 et M2, puis évalue le sixième corpus,
entièrement absent des données sources. M0 utilise le seuil natif {experiment.DEFAULT:.2f}.
M1 est SaT initial calibré ; M2 est le même SaT adapté par LoRA et calibré.
Les seuils ici ont été **resélectionnés depuis les prédictions déjà sauvegardées**
selon le F1 des segments avec espaces périphériques neutralisés, en moyenne
macro des cinq corpus de validation. Ils peuvent différer des seuils du rapport
cluster, choisis sur les segments stricts. Aucun modèle n'a été réentraîné.

Le F1 des frontières prend l'ensemble des débuts et fins internes des unités.
Chaque position est remplacée par le nombre de caractères non blancs qui la
précèdent ; les extrémités du récit sont exclues. Les segments sont comparés
par leurs deux offsets après retrait des espaces aux seuls bords. Précision,
rappel et F1 sont micro au sein d'un corpus cible, puis les six valeurs sont
moyennées sans pondération dans la ligne `macro_6_domains`.
Cette convention de frontière inclut les débuts **et** les fins de faits ;
elle n'est pas numériquement comparable au F1 des seules fins de phrases
rapporté par Frohmann et al. pour SaT.

Le rappel par rôle est la proportion d'unités de référence A0/A1/B/C dont
les deux bornes neutralisées sont retrouvées. SaT ne prédit aucun rôle.
Une fusion est un segment prédit contenant au moins deux unités de référence
complètes ; le taux principal est rapporté à tous les segments prédits.
Le taux secondaire `fusion_rate_supported` ne considère que les segments
contenant au moins une unité complète. Une fragmentation est une unité de
référence qui chevauche au moins deux segments prédits sur des caractères
non blancs ; son dénominateur est le nombre d'unités de référence.

Les IC à 95 % portent sur **M2 − M1** : bootstrap apparié de {n_bootstrap} tirages
de récits par corpus, seuils et modèles fixés, méthode percentile. L'IC macro
moyenne les gains de six rééchantillonnages indépendants, un par corpus.
Il ne couvre ni l'incertitude des annotations IA, ni celle due à la graine
d'entraînement ou au choix du checkpoint. {positive}/6 corpus ont un gain
positif en F1 des segments neutralisés.
Le seuil M1 atteint le bord inférieur de la grille testée (0,01) dans cinq
rotations : une grille plus fine pourrait modifier le choix. L'objectif
normalisé a été choisi après consultation des premiers résultats stricts ;
ce recalcul est donc exploratoire, pas une analyse confirmatoire prédéfinie.
Les six entraînements ont une seule graine et sauvegardent le dernier état,
pas le meilleur checkpoint de validation.

Fichiers : `tableau_principal.csv`, `tableau_1_segments.csv`, `tableau_2_frontieres.csv`,
`tableau_3_rappel_par_role.csv`, `tableau_4_fusions_fragmentations.csv`,
`gains_ic95.csv`, `seuils_retenus.csv`, `courbes_validation_sources.csv`,
deux figures PNG/PDF et `provenance.json`. Les tableaux sont des résultats
descriptifs sur préannotations IA ; il manque une validation humaine
indépendante pour en faire des performances finales d'article.
"""
    (out / "README.md").write_text(notes, encoding="utf-8")
    print(json.dumps({"output": str(out), "n_domains": len(experiment.CORPORA),
                      "macro_gain_span_f1": gain_rows[-1]["span_f1_gain"],
                      "positive_domains": positive}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path,
                        default=ROOT / "annotation/lodo_sat_transfer/sat12l_sm_lodo_v3")
    parser.add_argument("--runs", type=Path,
                        default=ROOT / "safer_sat_lodo/runs/sat12l_sm_lodo_v3")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "safer_sat_lodo/resultats_articles")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    generate(args.bundle.resolve(), args.runs.resolve(), args.out.resolve(),
             args.bootstrap, args.seed)


if __name__ == "__main__":
    main()
