#!/usr/bin/env python3
r"""Evaluate French PySBD against the same confirmed narrative_v3 gold as RQ1 M0.

From SAFER on Windows:
    .\text\.venv\Scripts\python.exe text\scripts\rq1_pysbd_baseline.py run
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.metadata
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RQ1 = ROOT / "annotation" / "rq1_sat_v3_20261017"
OUTPUT = ROOT / "annotation" / "rq1_pysbd_v3_20261017"
MODEL = "PySBD"
CORPORA = ("caou_chimie_plas", "btp", "metallurgie", "bois_textille",
           "nicollin", "service_temporaire")
SEED = 20261017
ROLES = ("A0", "A1", "B", "C", "AUTRE")
SCORE_FIELDS = ("span_tp", "span_pred", "span_ref", "boundary_tp",
                "boundary_pred", "boundary_ref")
ROLE_FIELDS = tuple(f"role_{role}_{suffix}" for role in ROLES for suffix in ("tp", "ref"))
DETAIL_FIELDS = (*SCORE_FIELDS, *ROLE_FIELDS, "fusion", "fragmented",
                 "gold_units", "pred_segments", "supported_pred_segments")


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def trim_span(raw: str, start: int, end: int) -> tuple[int, int]:
    while start < end and raw[start].isspace():
        start += 1
    while end > start and raw[end - 1].isspace():
        end -= 1
    return start, end


def normalized_spans(raw: str, spans: list[tuple[int, int]]) -> set[tuple[int, int]]:
    return {span for start, end in spans
            if (span := trim_span(raw, start, end))[0] < span[1]}


def nonspace_prefix(raw: str) -> list[int]:
    prefix = [0]
    for character in raw:
        prefix.append(prefix[-1] + (not character.isspace()))
    return prefix


def boundaries(prefix: list[int], spans: list[tuple[int, int]]) -> set[int]:
    total = prefix[-1]
    return {rank for start, end in spans for position in (start, end)
            if 0 < (rank := prefix[position]) < total}


def basic_score(row: dict, pred: list[tuple[int, int]]) -> dict[str, int]:
    raw = row["raw_text"]
    gold = [(unit["start"], unit["end"]) for unit in row["units"]]
    gold_spans = normalized_spans(raw, gold)
    pred_spans = normalized_spans(raw, pred)
    prefix = nonspace_prefix(raw)
    gold_boundaries = boundaries(prefix, gold)
    pred_boundaries = boundaries(prefix, pred)
    return {"span_tp": len(gold_spans & pred_spans), "span_pred": len(pred_spans),
            "span_ref": len(gold_spans),
            "boundary_tp": len(gold_boundaries & pred_boundaries),
            "boundary_pred": len(pred_boundaries), "boundary_ref": len(gold_boundaries)}


def full_score(row: dict, pred: list[tuple[int, int]]) -> dict[str, int]:
    raw = row["raw_text"]
    counts = collections.Counter(basic_score(row, pred))
    gold = normalized_spans(raw, [(unit["start"], unit["end"]) for unit in row["units"]])
    prediction = normalized_spans(raw, pred)
    prefix = nonspace_prefix(raw)
    pred_nonspace = [(prefix[start], prefix[end]) for start, end in prediction]
    for unit in row["units"]:
        role = unit["role"]
        counts[f"role_{role}_ref"] += 1
        counts[f"role_{role}_tp"] += (
            trim_span(raw, unit["start"], unit["end"]) in prediction)
        start, end = prefix[unit["start"]], prefix[unit["end"]]
        counts["fragmented"] += sum(max(start, a) < min(end, b)
                                    for a, b in pred_nonspace) >= 2
    for start, end in prediction:
        contained = sum(start <= a and b <= end for a, b in gold)
        counts["fusion"] += contained >= 2
        counts["supported_pred_segments"] += contained >= 1
    counts["gold_units"] = len(row["units"])
    counts["pred_segments"] = len(prediction)
    for field in DETAIL_FIELDS:
        counts[field] += 0
    return dict(counts)


def prf(tp: int, predicted: int, reference: int) -> tuple[float, float, float]:
    return (tp / predicted if predicted else 0.0,
            tp / reference if reference else 0.0,
            2 * tp / (predicted + reference) if predicted + reference else 0.0)


def aggregate_document_rows(rows: list[dict], target: str, model: str,
                            subset: str) -> dict:
    counts = collections.Counter()
    chosen = [row for row in rows if row["target"] == target and row["model"] == model
              and (subset == "all" or row["clean_reference"])]
    for row in chosen:
        counts.update({field: row.get(field, 0) for field in DETAIL_FIELDS})
    span_p, span_r, span_f = prf(counts["span_tp"], counts["span_pred"], counts["span_ref"])
    bound_p, bound_r, bound_f = prf(counts["boundary_tp"], counts["boundary_pred"],
                                   counts["boundary_ref"])
    return {"target": target, "model": model, "threshold": None, "subset": subset,
            "n_documents": len(chosen), **{field: counts[field] for field in SCORE_FIELDS},
            "span_precision": span_p, "span_recall": span_r, "span_f1": span_f,
            "boundary_precision": bound_p, "boundary_recall": bound_r,
            "boundary_f1": bound_f, "fusion_count": counts["fusion"],
            "supported_pred_segments": counts["supported_pred_segments"],
            "fusion_rate_supported": counts["fusion"] / counts["supported_pred_segments"]
            if counts["supported_pred_segments"] else 0.0,
            "fragmented_count": counts["fragmented"], "gold_units": counts["gold_units"],
            "fragmentation_rate": counts["fragmented"] / counts["gold_units"]
            if counts["gold_units"] else 0.0}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_gold(source: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    completed = json.loads((source / "completed.json").read_text(encoding="utf-8"))
    if (manifest["signature"]["reference"] != "narrative_v3 user-confirmed gold"
            or manifest["signature"]["snapshot_sha256"] != sha256(source / "gold_snapshot.jsonl")
            or completed["summary_sha256"] != sha256(source / "summary.json")):
        raise ValueError("RQ1 must be a complete, unchanged narrative_v3 gold run")
    with (source / "gold_snapshot.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    counts = collections.Counter(row["corpus"] for row in rows)
    if set(counts) != set(CORPORA) or len(rows) != 17927:
        raise ValueError("Gold snapshot has missing or unexpected documents")
    keys = {(row["corpus"], row["document_id"]) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("Duplicate gold document identifiers")
    return rows, manifest


def pysbd_spans(raw: str, segmenter) -> list[tuple[int, int]]:
    """Accept PySBD's own character offsets; never realign or invent a span."""
    spans = []
    cursor = 0
    for span in segmenter.segment(raw):
        start, end = span.start, span.end
        if (type(start) is not int or type(end) is not int
                or not 0 <= cursor <= start < end <= len(raw)
                or raw[start:end] != span.sent):
            raise ValueError("PySBD returned an invalid or nonliteral offset")
        if any(not char.isspace() for char in raw[cursor:start]):
            raise ValueError("PySBD left informative text between spans")
        spans.append((start, end))
        cursor = end
    if any(not char.isspace() for char in raw[cursor:]):
        raise ValueError("PySBD left informative text after its last span")
    return spans


def check_cached_spans(raw: str, spans: list[list[int]]) -> list[tuple[int, int]]:
    cursor = 0
    checked = []
    for item in spans:
        if (not isinstance(item, list) or len(item) != 2 or
                type(item[0]) is not int or type(item[1]) is not int):
            raise ValueError("Cached PySBD span has an invalid format")
        start, end = item
        if not 0 <= cursor <= start < end <= len(raw):
            raise ValueError("Cached PySBD spans overlap or exceed the narrative")
        if any(not char.isspace() for char in raw[cursor:start]):
            raise ValueError("Cached PySBD spans omit informative text")
        checked.append((start, end))
        cursor = end
    if any(not char.isspace() for char in raw[cursor:]):
        raise ValueError("Cached PySBD spans omit the narrative end")
    return checked


def cache_signature(source: Path) -> dict:
    return {"gold_snapshot_sha256": sha256(source / "gold_snapshot.jsonl"),
            "rq1_document_metrics_sha256": sha256(source / "document_metrics.csv"),
            "rq1_target_metrics_sha256": sha256(source / "target_metrics.csv"),
            "rq1_scoring_code_sha256": sha256(ROOT / "annotation" / "rq1_calibrate_sat_v3.py"),
            "script_sha256": sha256(Path(__file__)),
            "pysbd_version": importlib.metadata.version("pysbd"),
            "language": "fr", "clean": False, "char_span": True,
            "reference": "narrative_v3 user-confirmed gold"}


def open_cache(output: Path, signature: dict) -> sqlite3.Connection:
    output.mkdir(parents=True, exist_ok=True)
    config = output / "config.json"
    if config.exists():
        if json.loads(config.read_text(encoding="utf-8")) != signature:
            raise ValueError("Existing PySBD run belongs to different data or code")
    elif any(output.iterdir()):
        raise ValueError("Nonempty PySBD output directory has no config.json")
    else:
        write_json(config, signature)
    db = sqlite3.connect(output / "cache.sqlite3", timeout=60)
    db.execute("""CREATE TABLE IF NOT EXISTS predictions (
        corpus TEXT NOT NULL, document_id TEXT NOT NULL, text_sha256 TEXT NOT NULL,
        spans_json TEXT NOT NULL, PRIMARY KEY (corpus, document_id))""")
    db.commit()
    return db


def predict_all(rows: list[dict], db: sqlite3.Connection) -> None:
    import pysbd
    from tqdm import tqdm

    segmenter = pysbd.Segmenter(language="fr", clean=False, char_span=True)
    for corpus in CORPORA:
        group = [row for row in rows if row["corpus"] == corpus]
        existing = dict(db.execute(
            "SELECT document_id,text_sha256 FROM predictions WHERE corpus=?", (corpus,)))
        if len(existing) > len(group):
            raise ValueError(f"Unexpected cached documents for {corpus}")
        pending = []
        with tqdm(total=len(group), desc=f"PySBD {corpus}", unit="recit") as bar:
            for row in group:
                raw = row["raw_text"]
                doc_id = row["document_id"]
                text_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
                if doc_id in existing:
                    if existing[doc_id] != text_hash:
                        raise ValueError(f"Cached narrative text changed: {corpus}/{doc_id}")
                else:
                    try:
                        spans = pysbd_spans(raw, segmenter)
                    except ValueError as exc:
                        raise ValueError(f"Cannot evaluate {corpus}/{doc_id}: {exc}") from exc
                    pending.append((corpus, doc_id, text_hash, json.dumps(spans)))
                    if len(pending) >= 100:
                        with db:
                            db.executemany("INSERT INTO predictions VALUES (?,?,?,?)", pending)
                        pending.clear()
                bar.update(1)
            if pending:
                with db:
                    db.executemany("INSERT INTO predictions VALUES (?,?,?,?)", pending)
    if db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] != len(rows):
        raise ValueError("PySBD cache is incomplete")


def read_m0(source: Path, rows: list[dict]) -> tuple[dict, dict]:
    expected = {(row["corpus"], row["document_id"]) for row in rows}
    with (source / "document_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        original = [row for row in csv.DictReader(stream) if row["model"] == "M0"]
    found = {(row["target"], row["document_id"]): row for row in original}
    if len(original) != len(expected) or set(found) != expected:
        raise ValueError("M0 document IDs do not match the gold snapshot")
    with (source / "target_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        totals = {(row["target"], row["subset"]): row for row in csv.DictReader(stream)
                  if row["model"] == "M0"}
    if set(totals) != {(corpus, subset) for corpus in CORPORA
                      for subset in ("all", "clean_reference")}:
        raise ValueError("M0 target metrics are incomplete")
    return found, totals


def check_m0_parity(source: Path, rows: list[dict], totals: dict) -> None:
    """Replay RQ1 M0 predictions through these score rules before comparison."""
    from tqdm import tqdm

    lookup = {(row["corpus"], row["document_id"]): row for row in rows}
    for corpus in CORPORA:
        total = collections.Counter()
        seen = set()
        path = source / "predictions" / corpus / "M0.jsonl"
        with path.open(encoding="utf-8") as stream:
            for line in tqdm(stream, desc=f"Controle M0 {corpus}", unit="recit"):
                prediction = json.loads(line)
                key = corpus, prediction["document_id"]
                if key not in lookup or key in seen or prediction["corpus"] != corpus:
                    raise ValueError(f"M0 prediction has unexpected ID: {key}")
                seen.add(key)
                raw = lookup[key]["raw_text"]
                spans = []
                for item in prediction["segments"]:
                    start, end = item["start"], item["end"]
                    if raw[start:end] != item["text"]:
                        raise ValueError(f"M0 literal span differs: {key}")
                    spans.append([start, end])
                total.update(basic_score(lookup[key], check_cached_spans(raw, spans)))
        if seen != {key for key in lookup if key[0] == corpus}:
            raise ValueError(f"M0 prediction coverage differs for {corpus}")
        old = totals[corpus, "all"]
        if any(total[field] != int(old[field]) for field in SCORE_FIELDS):
            raise ValueError(f"Scoring convention no longer reproduces RQ1 M0: {corpus}")


def export_predictions(rows: list[dict], db: sqlite3.Connection,
                       output: Path) -> list[dict]:
    from tqdm import tqdm

    document_rows = []
    for corpus in CORPORA:
        group = [row for row in rows if row["corpus"] == corpus]
        path = output / "predictions" / f"{corpus}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in tqdm(group, desc=f"Scores {corpus}", unit="recit"):
                record = db.execute(
                    "SELECT text_sha256,spans_json FROM predictions WHERE corpus=? AND document_id=?",
                    (corpus, row["document_id"])).fetchone()
                if (record is None or record[0] !=
                        hashlib.sha256(row["raw_text"].encode("utf-8")).hexdigest()):
                    raise ValueError(f"Missing PySBD prediction: {corpus}/{row['document_id']}")
                spans = check_cached_spans(row["raw_text"], json.loads(record[1]))
                counts = full_score(row, spans)
                document_rows.append({"target": corpus, "document_id": row["document_id"],
                                      "model": MODEL,
                                      "clean_reference": not row["has_issues"] and
                                      not row["has_uncertain_units"], **counts})
                stream.write(json.dumps({"corpus": corpus, "document_id": row["document_id"],
                    "segments": [{"start": a, "end": b, "text": row["raw_text"][a:b]}
                                 for a, b in spans]}, ensure_ascii=False) + "\n")
        temporary.replace(path)
    return document_rows


def bootstrap_differences(pysbd_rows: list[dict], m0_rows: dict,
                          repetitions: int) -> list[dict]:
    import numpy as np
    from tqdm import tqdm

    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    output = []
    macro_draws = []
    for index, corpus in enumerate(tqdm(CORPORA, desc="Bootstrap", unit="corpus")):
        current = {row["document_id"]: row for row in pysbd_rows if row["target"] == corpus}
        ids = sorted(current)
        if ids != sorted(doc_id for (target, doc_id) in m0_rows if target == corpus):
            raise ValueError(f"M0/PySBD document pairing differs for {corpus}")
        arrays = {
            "M0": np.array([[int(m0_rows[corpus, doc_id][field]) for field in SCORE_FIELDS]
                            for doc_id in ids], dtype=np.int64),
            MODEL: np.array([[int(current[doc_id][field]) for field in SCORE_FIELDS]
                             for doc_id in ids], dtype=np.int64),
        }
        rng = np.random.default_rng(SEED + index)
        gains = np.empty((repetitions, 2), dtype=float)
        for repetition in range(repetitions):
            picks = rng.integers(0, len(ids), len(ids))
            left, right = (arrays[name][picks].sum(axis=0) for name in ("M0", MODEL))
            gains[repetition, 0] = prf(right[0], right[1], right[2])[2] - \
                prf(left[0], left[1], left[2])[2]
            gains[repetition, 1] = prf(right[3], right[4], right[5])[2] - \
                prf(left[3], left[4], left[5])[2]
        macro_draws.append(gains)
        for column, metric in enumerate(("span_f1", "boundary_f1")):
            output.append({"target": corpus, "metric": metric,
                           "ci95_low": float(np.quantile(gains[:, column], 0.025)),
                           "ci95_high": float(np.quantile(gains[:, column], 0.975)),
                           "replicates": repetitions})
    macro = np.mean(np.stack(macro_draws), axis=0)
    for column, metric in enumerate(("span_f1", "boundary_f1")):
        output.append({"target": "MACRO_6_CORPUS", "metric": metric,
                       "ci95_low": float(np.quantile(macro[:, column], 0.025)),
                       "ci95_high": float(np.quantile(macro[:, column], 0.975)),
                       "replicates": repetitions})
    return output


def write_results(output: Path, rows: list[dict],
                  document_rows: list[dict], m0_rows: dict, m0_totals: dict,
                  bootstrap: int, signature: dict) -> None:
    write_csv(output / "document_metrics.csv", document_rows,
                  ("target", "document_id", "model", "clean_reference", *DETAIL_FIELDS))
    metrics = [aggregate_document_rows(document_rows, corpus, MODEL, subset)
               for corpus in CORPORA for subset in ("all", "clean_reference")]
    write_csv(output / "target_metrics.csv", metrics,
                  ("target", "model", "threshold", "subset", "n_documents", *SCORE_FIELDS,
                   "span_precision", "span_recall", "span_f1", "boundary_precision",
                   "boundary_recall", "boundary_f1", "fusion_count", "supported_pred_segments",
                   "fusion_rate_supported", "fragmented_count", "gold_units", "fragmentation_rate"))
    recalls = []
    for corpus in CORPORA:
        current = [row for row in document_rows if row["target"] == corpus]
        for role in ROLES:
            matched = sum(row[f"role_{role}_tp"] for row in current)
            reference = sum(row[f"role_{role}_ref"] for row in current)
            recalls.append({"target": corpus, "model": MODEL, "role": role,
                            "matched": matched, "reference": reference,
                            "exact_recall": matched / reference if reference else 0.0})
    write_csv(output / "role_recall.csv", recalls,
                  ("target", "model", "role", "matched", "reference", "exact_recall"))

    comparisons = []
    for subset in ("all", "clean_reference"):
        for corpus in CORPORA:
            pysbd_metric = next(item for item in metrics
                                if item["target"] == corpus and item["subset"] == subset)
            old = m0_totals[corpus, subset]
            if int(old["n_documents"]) != pysbd_metric["n_documents"]:
                raise ValueError(f"M0/PySBD document counts differ for {corpus}/{subset}")
            comparisons.append({"target": corpus, "subset": subset,
                                "n_documents": pysbd_metric["n_documents"],
                                **{f"M0_{field}": float(old[field]) for field in (
                                    "span_precision", "span_recall", "span_f1",
                                    "boundary_precision", "boundary_recall", "boundary_f1")},
                                **{f"PySBD_{field}": pysbd_metric[field] for field in (
                                    "span_precision", "span_recall", "span_f1",
                                    "boundary_precision", "boundary_recall", "boundary_f1")},
                                "gain_span_f1": pysbd_metric["span_f1"] - float(old["span_f1"]),
                                "gain_boundary_f1": pysbd_metric["boundary_f1"] -
                                    float(old["boundary_f1"])})
        current = [row for row in comparisons if row["subset"] == subset]
        numeric = [name for name in current[0] if name.startswith(("M0_", "PySBD_", "gain_"))]
        comparisons.append({"target": "MACRO_6_CORPUS", "subset": subset,
                            "n_documents": sum(row["n_documents"] for row in current),
                            **{name: sum(row[name] for row in current) / len(CORPORA)
                               for name in numeric}})
    write_csv(output / "comparison_M0_PySBD.csv", comparisons, tuple(comparisons[0]))
    intervals = bootstrap_differences(document_rows, m0_rows, bootstrap)
    write_csv(output / "gain_ci95.csv", intervals,
                  ("target", "metric", "ci95_low", "ci95_high", "replicates"))
    write_json(output / "summary.json", {"reference": signature["reference"],
               "n_documents": len(rows), "pysbd_version": signature["pysbd_version"],
               "comparison": comparisons, "gain_ci95": intervals,
               "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    (output / "README.md").write_text(
        "# PySBD sur le gold narratif v3\n\n"
        "Même snapshot et mêmes conventions de score que RQ1 M0. "
        f"PySBD {signature['pysbd_version']} est configuré en français, "
        "sans nettoyage, avec offsets natifs.\n\n"
        "Le F1 de segments exacts neutralise uniquement les espaces périphériques. "
        "Le F1 de frontières exclut le début et la fin du récit et neutralise "
        "les espaces. Un score micro est calculé par corpus ; le macro est la "
        "moyenne des six scores. L'IC de gain est un bootstrap apparié par récit.\n\n"
        "Les annotations v3 ont été confirmées par l'utilisatrice, sans mesure "
        "indépendante d'accord entre annotateurs.\n",
        encoding="utf-8")
    files = {name: sha256(output / name) for name in
             ("document_metrics.csv", "target_metrics.csv", "role_recall.csv",
              "comparison_M0_PySBD.csv", "gain_ci95.csv", "summary.json")}
    write_json(output / "completed.json", {"config_sha256": sha256(output / "config.json"),
               "files": files, "n_documents": len(rows)})
    print(json.dumps({"comparison": str(output / "comparison_M0_PySBD.csv"),
                      "macro_all": next(item for item in comparisons
                                        if item["target"] == "MACRO_6_CORPUS" and
                                        item["subset"] == "all")}, ensure_ascii=False, indent=2))


def run(args: argparse.Namespace) -> None:
    source = Path(args.rq1).resolve()
    output = Path(args.output).resolve()
    rows, _ = read_gold(source)
    signature = cache_signature(source)
    m0_rows, m0_totals = read_m0(source, rows)
    check_m0_parity(source, rows, m0_totals)
    with closing(open_cache(output, signature)) as db:
        predict_all(rows, db)
        document_rows = export_predictions(rows, db, output)
    write_results(output, rows, document_rows, m0_rows, m0_totals,
                  args.bootstrap, signature)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("run", help="Evaluate French PySBD against RQ1 gold and M0")
    command.add_argument("--rq1", default=str(RQ1))
    command.add_argument("--output", default=str(OUTPUT))
    command.add_argument("--bootstrap", type=int, default=1000)
    command.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
