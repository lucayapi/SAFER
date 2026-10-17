#!/usr/bin/env python3
"""Explore M1 thresholds below .01 on LODO source validations only.

No target test file is opened, and no target score is computed. Predictions
are checkpointed per narrative so a cluster job can be resumed safely.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sqlite3
from pathlib import Path

from tqdm import tqdm

import sat_lodo
import sat_lodo_article_results as article

ROOT = Path(__file__).resolve().parents[2]
LOW = (0.001, 0.0025, 0.005, 0.0075, 0.01)
KEY = lambda value: f"{value:.4f}"


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def source_validation_rows(bundle: Path) -> tuple[dict, dict]:
    by_target, unique = {}, {}
    for target in sat_lodo.CORPORA:
        path = bundle / "folds" / f"target_{target}" / "val.jsonl"
        rows = sat_lodo.read_jsonl(path)
        if not rows or any(row["corpus"] == target for row in rows):
            raise ValueError(f"Source validation invalid for {target}")
        by_target[target] = rows
        for row in rows:
            key = (row["corpus"], row["document_id"])
            if key in unique and (unique[key]["raw_text"] != row["raw_text"] or
                                  unique[key]["units"] != row["units"]):
                raise ValueError(f"Nonidentical source validation row: {key}")
            unique[key] = row
    return by_target, unique


def cache_connection(path: Path, manifest_sha: str) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.execute("""CREATE TABLE IF NOT EXISTS predictions (
        corpus TEXT NOT NULL, document_id TEXT NOT NULL, raw_sha256 TEXT NOT NULL,
        spans_json TEXT NOT NULL, PRIMARY KEY(corpus, document_id))""")
    signature = json.dumps({"bundle_sha256": manifest_sha, "model": sat_lodo.MODEL,
                            "thresholds": LOW}, sort_keys=True)
    old = db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
    if old and old[0] != signature:
        db.close()
        raise ValueError("Existing low-threshold cache belongs to another bundle/grid")
    with db:
        db.execute("INSERT OR IGNORE INTO metadata VALUES('signature', ?)", (signature,))
    return db


def infer_base(unique: dict, db: sqlite3.Connection, runs: Path, *,
               max_documents: int | None = None) -> int:
    # A document can be source validation in several rotations. One base
    # inference is enough; selection below remains rotation-specific.
    existing = {(corpus, ident) for corpus, ident in
                db.execute("SELECT corpus, document_id FROM predictions")}
    pending = [row for key, row in unique.items() if key not in existing]
    if max_documents is not None:
        pending = pending[:max_documents]
    if not pending:
        return 0
    model = sat_lodo.load_sat()
    checked = 0
    old_caches = {}
    try:
        for row in tqdm(pending, desc="SaT base, validation source", unit="récit"):
            raw = row["raw_text"]
            probabilities = list(model.predict_proba([raw], stride=64))[0]
            spans = {KEY(t): sat_lodo.spans_from_probabilities(raw, probabilities, t)
                     for t in LOW}
            # Compare .01 against one existing source-validation cache. This
            # guards the new pass against a changed model or split procedure.
            for target in sat_lodo.CORPORA:
                if target == row["corpus"]:
                    continue
                path = runs / f"target_{target}" / "predictions.sqlite3"
                if not path.is_file():
                    continue
                if target not in old_caches:
                    old_caches[target] = sqlite3.connect(
                        f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
                found = old_caches[target].execute(
                    "SELECT spans_json FROM predictions WHERE model='base' AND corpus=? AND document_id=?",
                    (row["corpus"], row["document_id"]),
                ).fetchone()
                if found:
                    previous = json.loads(found[0])["0.01"]
                    if spans[KEY(.01)] != [tuple(pair) for pair in previous]:
                        raise ValueError(f"Cached .01 prediction differs: {row['corpus']}/{row['document_id']}")
                    checked += 1
                    break
            else:
                raise ValueError(f"No original source-validation cache: {row['corpus']}/{row['document_id']}")
            digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            with db:
                db.execute("INSERT INTO predictions VALUES(?,?,?,?)",
                           (row["corpus"], row["document_id"], digest, json.dumps(spans)))
    finally:
        for old in old_caches.values():
            old.close()
    return checked


def source_curves(by_target: dict, db: sqlite3.Connection) -> list[dict]:
    curves = []
    for target, rows in by_target.items():
        counts = collections.defaultdict(collections.Counter)
        for row in rows:
            found = db.execute(
                "SELECT raw_sha256, spans_json FROM predictions WHERE corpus=? AND document_id=?",
                (row["corpus"], row["document_id"]),
            ).fetchone()
            if not found:
                raise ValueError(f"Uncomputed source validation: {target}/{row['document_id']}")
            if found[0] != hashlib.sha256(row["raw_text"].encode("utf-8")).hexdigest():
                raise ValueError(f"Changed source text: {target}/{row['document_id']}")
            pred = json.loads(found[1])
            gold = [(unit["start"], unit["end"]) for unit in row["units"]]
            for threshold in LOW:
                tp, n_pred, n_ref = article.score_span(
                    row["raw_text"], gold, pred[KEY(threshold)])
                counts[threshold, row["corpus"]].update(
                    tp=tp, pred=n_pred, ref=n_ref)
        for threshold in LOW:
            domains = [domain for domain in sat_lodo.CORPORA if domain != target]
            per_domain = {}
            for domain in domains:
                c = counts[threshold, domain]
                if not c["ref"]:
                    raise ValueError(f"No reference facts in {target}/{domain}")
                per_domain[domain] = article.prf(c["tp"], c["pred"], c["ref"])[2]
            curves.append({"target": target, "model": "base",
                           "threshold": threshold,
                           "source_macro_span_f1_trimmed": sum(per_domain.values()) / 5,
                           **{f"f1_{domain}": per_domain.get(domain)
                              for domain in sat_lodo.CORPORA}})
    return curves


def cached_original_curves(by_target: dict, runs: Path) -> list[dict]:
    curves = []
    for target, rows in by_target.items():
        path = runs / f"target_{target}" / "predictions.sqlite3"
        if not path.is_file():
            raise FileNotFoundError(path)
        counts = collections.defaultdict(collections.Counter)
        with sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True) as db:
            for row in rows:
                cached = article.load_predictions(db, "base", row)
                gold = [(unit["start"], unit["end"]) for unit in row["units"]]
                for threshold in sat_lodo.THRESHOLDS:
                    tp, n_pred, n_ref = article.score_span(
                        row["raw_text"], gold, cached[f"{threshold:.2f}"])
                    counts[threshold, row["corpus"]].update(
                        tp=tp, pred=n_pred, ref=n_ref)
        for threshold in sat_lodo.THRESHOLDS:
            domains = [domain for domain in sat_lodo.CORPORA if domain != target]
            per_domain = {domain: article.prf(
                counts[threshold, domain]["tp"], counts[threshold, domain]["pred"],
                counts[threshold, domain]["ref"])[2] for domain in domains}
            curves.append({"target": target, "model": "base", "threshold": threshold,
                           "source_macro_span_f1_trimmed": sum(per_domain.values()) / 5,
                           **{f"f1_{domain}": per_domain.get(domain)
                              for domain in sat_lodo.CORPORA}})
    return curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", choices=["run"])
    parser.add_argument("--bundle", type=Path, default=ROOT / "annotation" /
                        "lodo_sat_transfer" / "sat12l_sm_lodo_v3")
    parser.add_argument("--runs", type=Path, default=ROOT / "safer_sat_lodo" /
                        "runs" / "sat12l_sm_lodo_v3")
    parser.add_argument("--out", type=Path, default=ROOT / "safer_sat_lodo" /
                        "resultats_articles" / "validation_seuils_inferieurs")
    parser.add_argument("--max-documents", type=int, default=None,
                        help="Only for a smoke check; does not produce final curves")
    args = parser.parse_args()
    if args.max_documents is not None and args.max_documents < 1:
        parser.error("--max-documents must be positive")
    for path in (args.bundle, args.runs):
        if not path.is_dir():
            parser.error(f"Missing input directory: {path}")
    manifest_sha = article.sha256(args.bundle / "manifest.json")
    for target in sat_lodo.CORPORA:
        binding = args.runs / f"target_{target}" / "bundle_binding.json"
        if json.loads(binding.read_text(encoding="utf-8"))["bundle_manifest_sha256"] != manifest_sha:
            raise ValueError(f"Run and bundle differ: {target}")
    by_target, unique = source_validation_rows(args.bundle)
    args.out.mkdir(parents=True, exist_ok=True)
    with cache_connection(args.out / "predictions_validation.sqlite3", manifest_sha) as db:
        checked = infer_base(unique, db, args.runs, max_documents=args.max_documents)
        completed = db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        print(f"Validation source: {completed}/{len(unique)} récits distincts; "
              f"{checked} nouveaux contrôles d'équivalence à 0,01", flush=True)
        if completed < len(unique):
            print("Contrôle partiel : aucune courbe finale écrite.", flush=True)
            return
        curves = source_curves(by_target, db)
    old_curves = cached_original_curves(by_target, args.runs)
    for fresh in curves:
        if fresh["threshold"] == .01:
            old = next(row for row in old_curves if row["target"] == fresh["target"] and
                       row["threshold"] == .01)
            if abs(fresh["source_macro_span_f1_trimmed"] -
                   old["source_macro_span_f1_trimmed"]) > 1e-12:
                raise ValueError(f"Original and recalculated .01 scores differ: {fresh['target']}")
    selection = []
    for target in sat_lodo.CORPORA:
        candidates = [(row["threshold"], row["source_macro_span_f1_trimmed"])
                      for row in old_curves if row["target"] == target and
                      row["threshold"] > .01]
        candidates.extend((row["threshold"], row["source_macro_span_f1_trimmed"])
                          for row in curves if row["target"] == target)
        winner = min(candidates, key=lambda x: (-x[1], abs(x[0] - sat_lodo.DEFAULT), x[0]))
        old = next(row for row in old_curves if row["target"] == target and
                   row["threshold"] == .01)
        selection.append({"target": target, "previous_lower_bound": .01,
                          "f1_at_0_01": old["source_macro_span_f1_trimmed"],
                          "selected_threshold_source_only": winner[0],
                          "selected_source_macro_f1": winner[1],
                          "delta_vs_0_01": winner[1] - old["source_macro_span_f1_trimmed"],
                          "n_source_validation": len(by_target[target])})
    fields = ["target", "model", "threshold", "source_macro_span_f1_trimmed"] + [
        f"f1_{domain}" for domain in sat_lodo.CORPORA]
    write_csv(args.out / "courbes_seuils_inferieurs.csv", curves, fields)
    write_csv(args.out / "courbes_grille_elargie.csv", old_curves + [
        row for row in curves if row["threshold"] < .01], fields)
    write_csv(args.out / "selection_validation_sources.csv", selection, list(selection[0]))
    (args.out / "provenance.json").write_text(json.dumps({
        "reference": "AI_DIRECT_PREANNOTATIONS_NOT_HUMAN_GOLD",
        "bundle_manifest_sha256": manifest_sha, "model": sat_lodo.MODEL,
        "new_thresholds": LOW, "selection": "source macro of per-domain micro trimmed-span F1",
        "target_test_used": False, "n_unique_validation_narratives": len(unique),
        "val_jsonl_sha256": {target: article.sha256(
            args.bundle / "folds" / f"target_{target}" / "val.jsonl")
            for target in sat_lodo.CORPORA},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in selection:
        print(f"{row['target']}: {row['selected_threshold_source_only']:.4f} "
              f"(delta vs .01 = {row['delta_vs_0_01']:+.6f})", flush=True)
    print(f"Source-only report: {args.out}", flush=True)


if __name__ == "__main__":
    main()
