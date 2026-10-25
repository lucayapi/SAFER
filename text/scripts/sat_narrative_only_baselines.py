#!/usr/bin/env python3
"""Run and compare M0-M2 on the shared narrative-only LODO folds.

M0 is French PySBD, M1 is pinned pretrained SaT at its default threshold, and
M2 is the same SaT with its threshold selected on the five source validations.
All scores reuse the edge-normalized exact-span metric implemented by the M3
runner. The ``compare`` command refuses to combine results unless M3's run
binding, test IDs, reference counts, seed, and normalization match the bundle.
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
from pathlib import Path

import sat_narrative_only_lodo as m3
import sat_narrative_only_runtime as runtime

ROOT = runtime.ROOT
PREPARED = ROOT / "annotation" / "lodo_sat_transfer" / "narrative_only_lodo_v1"
DEFAULT_OUTPUT = ROOT / "safer_sat_lodo" / "runs" / "narrative_only_lodo_v3_baselines"
VERSION = "sat_narrative_only_baselines_v3"
METHODS = ("M0", "M1", "M2")
NORMALIZATION = "trim Unicode whitespace and punctuation at span edges only"
METRIC = "edge_normalized_exact_f1"
COUNTS = ("span_tp", "span_pred", "span_ref")


def digest(path: Path) -> str:
    return runtime.digest(path)


def code_digest() -> str:
    h = hashlib.sha256()
    for path in (Path(__file__), Path(m3.__file__), Path(runtime.__file__)):
        h.update(path.name.encode("utf-8"))
        h.update(path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8"))
    return h.hexdigest()


def binding(prepared: Path) -> dict:
    manifest = m3.check_prepared(prepared)
    pinned = json.loads((prepared / "pinned_runtime.json").read_text(encoding="utf-8"))
    return {
        "version": VERSION,
        "prepared_manifest_sha256": digest(prepared / "manifest.json"),
        "baseline_code_sha256": code_digest(),
        "m3_binding": m3.binding(prepared),
        "model_revision": pinned["model_revision"],
        "tokenizer_revision": pinned["tokenizer_revision"],
        "fold_order": manifest["fold_order"],
        "seed": runtime.SEED,
        "normalization": NORMALIZATION,
        "thresholds": list(runtime.THRESHOLDS),
        "default_threshold": runtime.DEFAULT,
    }


def write_json(path: Path, value: object) -> None:
    runtime.write_json(path, value)


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    runtime.write_csv(path, rows, fields)


def load_all_test_rows(prepared: Path) -> list[dict]:
    rows = []
    seen = set()
    for target in runtime.CORPORA:
        fold = prepared / "folds" / f"target_{target}"
        test = runtime.read_jsonl(fold / "test.jsonl")
        if not test or any(row["corpus"] != target for row in test):
            raise ValueError(f"Invalid or empty held-out test fold: {target}")
        for row in test:
            key = row["corpus"], row["document_id"]
            if key in seen:
                raise ValueError(f"Duplicate held-out document across folds: {key}")
            seen.add(key)
            rows.append(row)
    dataset_manifest = json.loads((prepared / "source_dataset_manifest.json").read_text(encoding="utf-8"))
    if len(rows) != dataset_manifest["total_documents"]:
        raise ValueError("Held-out test folds do not cover the complete narrative-only dataset")
    return rows


def load_probability_rows(prepared: Path, test_rows: list[dict]) -> list[dict]:
    """Unique test and source-validation narratives required by M1/M2."""
    unique: dict[tuple[str, str], dict] = {}
    for row in test_rows:
        unique[(row["corpus"], row["document_id"])] = row
    for target in runtime.CORPORA:
        fold = prepared / "folds" / f"target_{target}"
        for row in runtime.read_jsonl(fold / "val.jsonl"):
            key = row["corpus"], row["document_id"]
            previous = unique.get(key)
            if previous is not None and previous["raw_text"] != row["raw_text"]:
                raise ValueError(f"Conflicting text for shared probability-cache ID: {key}")
            unique.setdefault(key, row)
    # Empty texts are deliberately not sent to SaT or stored in its cache.
    return [row for row in unique.values() if row["raw_text"].strip()]


def read_cached_spans(raw: str, value: str) -> list[tuple[int, int]]:
    spans = json.loads(value)
    checked = []
    cursor = 0
    for item in spans:
        if (not isinstance(item, list) or len(item) != 2 or
                type(item[0]) is not int or type(item[1]) is not int):
            raise ValueError("Cached span has an invalid format")
        start, end = item
        if not 0 <= cursor <= start < end <= len(raw):
            raise ValueError("Cached spans overlap or exceed the narrative")
        if any(not char.isspace() for char in raw[cursor:start]):
            raise ValueError("Cached spans omit informative text")
        checked.append((start, end))
        cursor = end
    if any(not char.isspace() for char in raw[cursor:]):
        raise ValueError("Cached spans omit informative text at the narrative end")
    return checked


def pysbd_spans(raw: str, segmenter) -> list[tuple[int, int]]:
    spans = []
    cursor = 0
    for span in segmenter.segment(raw):
        start, end = span.start, span.end
        if (type(start) is not int or type(end) is not int or
                not 0 <= cursor <= start < end <= len(raw) or raw[start:end] != span.sent):
            raise ValueError("PySBD returned an invalid or nonliteral character span")
        if any(not char.isspace() for char in raw[cursor:start]):
            raise ValueError("PySBD omitted informative text")
        spans.append((start, end))
        cursor = end
    if any(not char.isspace() for char in raw[cursor:]):
        raise ValueError("PySBD omitted informative text at the narrative end")
    return spans


def open_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS predictions (corpus TEXT, document_id TEXT, "
                "text_sha256 TEXT, spans_json TEXT, PRIMARY KEY(corpus,document_id))")
    db.commit()
    return db


def cache_pysbd(rows: list[dict], db: sqlite3.Connection) -> None:
    import pysbd
    from tqdm import tqdm

    segmenter = pysbd.Segmenter(language="fr", clean=False, char_span=True)
    present = {(c, ident): text_hash for c, ident, text_hash in db.execute(
        "SELECT corpus,document_id,text_sha256 FROM predictions")}
    if len(present) > len(rows):
        raise ValueError("PySBD cache contains unexpected documents")
    for row in tqdm(rows, desc="PySBD M0", unit="narrative"):
        key = row["corpus"], row["document_id"]
        text_hash = hashlib.sha256(row["raw_text"].encode("utf-8")).hexdigest()
        if key in present:
            if present[key] != text_hash:
                raise ValueError(f"Cached PySBD text changed: {key}")
            continue
        spans = pysbd_spans(row["raw_text"], segmenter) if row["raw_text"].strip() else []
        with db:
            db.execute("INSERT INTO predictions VALUES (?,?,?,?)",
                       (*key, text_hash, json.dumps(spans)))
    if db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] != len(rows):
        raise ValueError("PySBD cache is incomplete")


def get_pysbd(db: sqlite3.Connection, row: dict) -> list[tuple[int, int]]:
    record = db.execute("SELECT text_sha256,spans_json FROM predictions "
                        "WHERE corpus=? AND document_id=?",
                        (row["corpus"], row["document_id"])).fetchone()
    expected = hashlib.sha256(row["raw_text"].encode("utf-8")).hexdigest()
    if record is None or record[0] != expected:
        raise ValueError(f"Missing or stale PySBD prediction: {row['corpus']}/{row['document_id']}")
    return read_cached_spans(row["raw_text"], record[1])


def init_output(output: Path, expected: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "binding.json"
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Output belongs to a different protocol: {output}")
    elif any(output.iterdir()):
        raise ValueError(f"Refusing nonempty unbound output directory: {output}")
    else:
        write_json(marker, expected)


def write_prediction_file(path: Path, rows: list[dict], spans: dict[str, list[tuple[int, int]]],
                          threshold: float | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps({"corpus": row["corpus"], "document_id": row["document_id"],
                "threshold": threshold, "segments": [{"start": a, "end": b}
                for a, b in spans[row["document_id"]]]}, ensure_ascii=False) + "\n")
    temporary.replace(path)


def counts_for(rows: list[dict], span_by_id: dict[str, list[tuple[int, int]]]) -> dict:
    totals = collections.Counter()
    for row in rows:
        totals.update(m3.score(row, span_by_id[row["document_id"]]))
    p, r, f1 = m3.prf(totals)
    return {**{key: int(totals[key]) for key in COUNTS},
            "precision": p, "recall": r, METRIC: f1}


def run(args: argparse.Namespace) -> None:
    prepared, output = args.prepared.resolve(), args.output.resolve()
    m3.check_prepared(prepared)
    expected_binding = binding(prepared)
    init_output(output, expected_binding)
    rows = load_all_test_rows(prepared)
    probability_rows = load_probability_rows(prepared, rows)
    model_dir, tokenizer_dir = m3.pinned_backbone(prepared)
    model = runtime.load_pinned_sat(model_dir, tokenizer_dir)
    prob_db_path = output / "pretrained_probabilities.sqlite3"
    pysbd_db_path = output / "pysbd_predictions.sqlite3"
    with closing(open_cache(pysbd_db_path)) as pysbd_db:
        cache_pysbd(rows, pysbd_db)
        with closing(runtime._probability_connection(prob_db_path, create=True)) as prob_db:
            runtime.infer_adapted(model, prob_db, probability_rows)
            expected_probability_ids = {(r["corpus"], r["document_id"])
                                        for r in probability_rows}
            actual_probability_ids = {(c, ident) for c, ident in prob_db.execute(
                "SELECT corpus,document_id FROM probabilities")}
            if actual_probability_ids != expected_probability_ids:
                raise ValueError("Pretrained SaT probability cache has unexpected or missing documents")
            runtime.verify_native(model, prob_db, probability_rows, runtime.DEFAULT)
            metric_rows = []
            threshold_rows = []
            for target in runtime.CORPORA:
                fold = prepared / "folds" / f"target_{target}"
                val = runtime.read_jsonl(fold / "val.jsonl")
                test = runtime.read_jsonl(fold / "test.jsonl")
                # Ignore whitespace-only records just as inference does, while
                # retaining their corpus labels for the five-source check.
                val_for_threshold = [
                    row if row["raw_text"].strip() else {**row, "raw_text": ""}
                    for row in val
                ]
                chosen, curve = m3.select_threshold(val_for_threshold, prob_db)
                for point in curve:
                    threshold_rows.append({"target": target, **point})
                variants = {}
                variants["M0"] = {row["document_id"]: get_pysbd(pysbd_db, row) for row in test}
                variants["M1"] = {row["document_id"]: (
                    runtime.legacy.spans_from_probabilities(row["raw_text"],
                        runtime.get_probs(prob_db, row), runtime.DEFAULT) if row["raw_text"].strip() else [])
                    for row in test}
                variants["M2"] = {row["document_id"]: (
                    runtime.legacy.spans_from_probabilities(row["raw_text"],
                        runtime.get_probs(prob_db, row), chosen) if row["raw_text"].strip() else [])
                    for row in test}
                for method in METHODS:
                    pred = variants[method]
                    write_prediction_file(output / "predictions" / target / f"{method}.jsonl",
                                          test, pred,
                                          None if method == "M0" else
                                          runtime.DEFAULT if method == "M1" else chosen)
                    metric_rows.append({"target": target, "method": method, "n_test": len(test),
                        "threshold": None if method == "M0" else
                        runtime.DEFAULT if method == "M1" else chosen,
                        **counts_for(test, pred)})
        model = None
    by_method = {method: [r for r in metric_rows if r["method"] == method] for method in METHODS}
    for method in METHODS:
        if len(by_method[method]) != len(runtime.CORPORA):
            raise ValueError(f"Incomplete target metrics for {method}")
        scores = [r[METRIC] for r in by_method[method]]
        metric_rows.append({"target": "MACRO_6_CORPUS", "method": method,
            "n_test": sum(r["n_test"] for r in by_method[method]), "threshold": "per protocol",
            METRIC: sum(scores) / len(scores),
            "precision": sum(r["precision"] for r in by_method[method]) / len(by_method[method]),
            "recall": sum(r["recall"] for r in by_method[method]) / len(by_method[method]),
            **{key: sum(r[key] for r in by_method[method]) for key in COUNTS}})
    fields = ("target", "method", "n_test", "threshold", *COUNTS,
              "precision", "recall", METRIC)
    write_csv(output / "baseline_metrics.csv", metric_rows, fields)
    write_csv(output / "source_thresholds.csv", threshold_rows,
              ("target", "threshold", "source_macro_edge_normalized_exact_f1",
               *(f"f1_{corpus}" for corpus in runtime.CORPORA), "selected"))
    result_files = [output / "baseline_metrics.csv", output / "source_thresholds.csv",
                    *sorted((output / "predictions").rglob("*.jsonl"))]
    files = {path.relative_to(output).as_posix(): digest(path) for path in result_files}
    output_manifest = {"version": VERSION, "binding": expected_binding,
        "pysbd_version": importlib.metadata.version("pysbd"), "files": files,
        "completed_at_utc": runtime.stamp()}
    write_json(output / "manifest.json", output_manifest)
    write_json(output / "complete.json", {"manifest_sha256": digest(output / "manifest.json")})
    print(json.dumps({"baseline_root": str(output),
        "macro_f1": {row["method"]: row[METRIC] for row in metric_rows
                     if row["target"] == "MACRO_6_CORPUS"}}, ensure_ascii=False, indent=2))


def read_prediction_file(path: Path, expected_rows: list[dict], method: str) -> dict:
    expected = {(row["corpus"], row["document_id"]): row for row in expected_rows}
    predictions = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            key = item.get("corpus"), item.get("document_id")
            if key not in expected or key in predictions:
                raise ValueError(f"{method} predictions contain unexpected/duplicate ID: {key}")
            spans = []
            raw = expected[key]["raw_text"]
            for span in item.get("segments", []):
                a, b = span.get("start"), span.get("end")
                if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(raw):
                    raise ValueError(f"Invalid {method} span offsets for {key}")
                spans.append((a, b))
            predictions[key] = spans
    if set(predictions) != set(expected):
        raise ValueError(f"{method} predictions do not cover the held-out documents")
    return predictions


def compare(args: argparse.Namespace) -> None:
    prepared, baseline_root, m3_root, output = (args.prepared.resolve(), args.baseline_root.resolve(),
        args.m3_root.resolve(), args.output.resolve())
    m3.check_prepared(prepared)
    expected_m3_binding = m3.binding(prepared)
    baseline_manifest = json.loads((baseline_root / "manifest.json").read_text(encoding="utf-8"))
    baseline_complete = json.loads((baseline_root / "complete.json").read_text(encoding="utf-8"))
    if (baseline_manifest.get("version") != VERSION or
            baseline_manifest.get("binding") != binding(prepared) or
            baseline_complete.get("manifest_sha256") != digest(baseline_root / "manifest.json")):
        raise ValueError("M0-M2 results do not match the shared narrative-only protocol")
    for relative, sha in baseline_manifest.get("files", {}).items():
        member = (baseline_root / relative).resolve()
        if (not member.is_relative_to(baseline_root) or not member.is_file() or digest(member) != sha):
            raise ValueError(f"M0-M2 result file missing or changed: {relative}")
    m3_summary = json.loads((m3_root / "summary.json").read_text(encoding="utf-8"))
    if (m3_summary.get("version") != m3.VERSION or
            m3_summary.get("binding") != expected_m3_binding or
            m3_summary.get("normalization") != NORMALIZATION or
            m3_summary.get("seed") != runtime.SEED):
        raise ValueError("M3 summary is not bound to these folds, seed, and metric")
    output.mkdir(parents=True, exist_ok=True)
    table = []
    for target in runtime.CORPORA:
        test = runtime.read_jsonl(prepared / "folds" / f"target_{target}" / "test.jsonl")
        if not test or any(row["corpus"] != target for row in test):
            raise ValueError(f"Invalid test data for {target}")
        expected_ref = sum(m3.score(row, [])["span_ref"] for row in test)
        metrics = {}
        for method in METHODS:
            pred = read_prediction_file(baseline_root / "predictions" / target / f"{method}.jsonl",
                                        test, method)
            counts = collections.Counter()
            for row in test:
                counts.update(m3.score(row, pred[row["corpus"], row["document_id"]]))
            p, r, f1 = m3.prf(counts)
            if counts["span_ref"] != expected_ref:
                raise ValueError(f"Reference counts differ for {target}/{method}")
            metrics[method] = {**counts, "precision": p, "recall": r, METRIC: f1}
        run = m3_root / f"target_{target}"
        run_binding = json.loads((run / "binding.json").read_text(encoding="utf-8"))
        result = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        if (run_binding != expected_m3_binding or result.get("version") != m3.VERSION or
                result.get("target") != target or result.get("seed") != runtime.SEED or
                result.get("normalization") != NORMALIZATION or result.get("n_test") != len(test) or
                result.get("counts", {}).get("span_ref") != expected_ref):
            raise ValueError(f"M3 fold metadata/counts mismatch for {target}")
        m3_predictions = read_prediction_file(run / "predictions.jsonl", test, "M3")
        m3_counts = collections.Counter()
        for row in test:
            m3_counts.update(m3.score(row, m3_predictions[row["corpus"], row["document_id"]]))
        p, r, f1 = m3.prf(m3_counts)
        if (m3_counts["span_ref"] != expected_ref or m3_counts["span_tp"] != result["counts"]["span_tp"] or
                m3_counts["span_pred"] != result["counts"]["span_pred"] or
                abs(f1 - result["metrics"][METRIC]) > 1e-12):
            raise ValueError(f"M3 predictions do not reproduce its stored scores: {target}")
        metrics["M3"] = {**m3_counts, "precision": p, "recall": r, METRIC: f1}
        for method, values in metrics.items():
            table.append({"target": target, "n_test": len(test), "method": method,
                          "precision": values["precision"], "recall": values["recall"],
                          METRIC: values[METRIC], **{k: int(values[k]) for k in COUNTS}})
    for method in ("M0", "M1", "M2", "M3"):
        target_rows = [row for row in table if row["method"] == method]
        if len(target_rows) != len(runtime.CORPORA):
            raise ValueError(f"Comparison is missing target folds for {method}")
        table.append({"target": "MACRO_6_CORPUS", "n_test": sum(r["n_test"] for r in target_rows),
            "method": method, "precision": sum(r["precision"] for r in target_rows) / 6,
            "recall": sum(r["recall"] for r in target_rows) / 6,
            METRIC: sum(r[METRIC] for r in target_rows) / 6,
            **{k: sum(r[k] for r in target_rows) for k in COUNTS}})
    fields = ("target", "n_test", "method", *COUNTS, "precision", "recall", METRIC)
    write_csv(output / "comparison_M0_M3.csv", table, fields)
    comparison_manifest = {"version": "narrative_only_M0_M3_comparison_v1",
        "prepared_manifest_sha256": digest(prepared / "manifest.json"),
        "baseline_manifest_sha256": digest(baseline_root / "manifest.json"),
        "m3_summary_sha256": digest(m3_root / "summary.json"),
        "normalization": NORMALIZATION,
        "macro_f1": {method: next(r[METRIC] for r in table
                                  if r["target"] == "MACRO_6_CORPUS" and r["method"] == method)
                     for method in ("M0", "M1", "M2", "M3")}}
    write_json(output / "comparison_manifest.json", comparison_manifest)
    print(json.dumps(comparison_manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--prepared", type=Path, default=PREPARED)
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_parser.set_defaults(func=run)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--prepared", type=Path, default=PREPARED)
    compare_parser.add_argument("--baseline-root", type=Path, default=DEFAULT_OUTPUT)
    compare_parser.add_argument("--m3-root", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.set_defaults(func=compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
