"""Build the six fact-level text datasets from the frozen RQ1 narrative_v3 snapshot.

The operation validates every source span and stages every CSV before replacing
any installed dataset. Existing datasets are copied to a private legacy archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SAFER = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = SAFER / "annotation" / "rq1_sat_v3_20261017"
SNAPSHOT = SNAPSHOT_DIR / "gold_snapshot.jsonl"
SOURCE_MANIFEST = SNAPSHOT_DIR / "manifest.json"
DATASET = SAFER / "text" / "dataset"
EXPECTED_SHA = "4845dbec927c5cd1161360729382ac5497c3f5c41d1562b03d5020c2f20bdca8"
EXPECTED_DOCUMENTS = 17927
EXPECTED_UNITS = 147487
CORPORA = {
    "btp": "btp",
    "metallurgie": "metallurgie",
    "caou_chimie_plas": "caou",
    "bois_textille": "bois_textille",
    "service_temporaire": "service_temporaire",
    "nicollin": "nicollin",
}
ROLES = {"A0", "A1", "B", "C", "AUTRE"}
FIELDS = [
    "accident_id", "fact_id", "sentence", "accident_summary", "pred_label",
    "pred_ok", "doc_id", "start", "end", "uncertain", "validation_method",
    "gold_version",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_rows(snapshot: Path, source_manifest: Path, *, expected_sha: str = EXPECTED_SHA):
    actual_sha = sha256(snapshot)
    if actual_sha != expected_sha:
        raise ValueError(f"RQ1 snapshot SHA256 mismatch: {actual_sha}")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if manifest["signature"]["snapshot_sha256"] != actual_sha:
        raise ValueError("RQ1 source manifest does not match the snapshot")

    document_ids: dict[str, set[str]] = defaultdict(set)
    document_counts: Counter[str] = Counter()
    unit_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    with snapshot.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            doc = json.loads(line)
            corpus = doc["corpus"]
            if corpus not in CORPORA:
                raise ValueError(f"Unknown corpus on line {line_no}: {corpus}")
            alias = CORPORA[corpus]
            accident_id = str(doc["document_id"])
            if not accident_id or accident_id in document_ids[alias]:
                raise ValueError(f"Missing/duplicate document id: {alias}/{accident_id}")
            document_ids[alias].add(accident_id)
            document_counts[alias] += 1
            text = doc["raw_text"]
            if not isinstance(text, str) or not text:
                raise ValueError(f"Empty text: {alias}/{accident_id}")
            units = doc["units"]
            if not units:
                yield alias, None, accident_id
            previous_end = 0
            for fact_id, unit in enumerate(units, 1):
                start, end = unit["start"], unit["end"]
                if not isinstance(start, int) or not isinstance(end, int) or not (previous_end <= start < end <= len(text)):
                    raise ValueError(f"Invalid/overlapping offsets: {alias}/{accident_id}/{fact_id}")
                previous_end = end
                sentence = text[start:end]
                if not sentence.strip() or unit["role"] not in ROLES:
                    raise ValueError(f"Invalid span/role: {alias}/{accident_id}/{fact_id}")
                unit_counts[alias] += 1
                role_counts[unit["role"]] += 1
                yield alias, {
                    "accident_id": accident_id,
                    "fact_id": fact_id,
                    "sentence": sentence,
                    "accident_summary": text,
                    "pred_label": unit["role"],
                    "pred_ok": True,
                    "doc_id": unit_counts[alias],
                    "start": start,
                    "end": end,
                    "uncertain": bool(unit.get("uncertain", False)),
                    "validation_method": doc.get("validation_method", ""),
                    "gold_version": "narrative_v3",
                }, accident_id
    expected = manifest["signature"]["source"]
    for original, alias in CORPORA.items():
        if document_counts[alias] != expected[original]["validated"]:
            raise ValueError(f"Document count mismatch: {alias}")
    if sum(document_counts.values()) != EXPECTED_DOCUMENTS or sum(unit_counts.values()) != EXPECTED_UNITS:
        raise ValueError(f"Gold totals mismatch: docs={sum(document_counts.values())}, units={sum(unit_counts.values())}")
    if set(role_counts) != ROLES:
        raise ValueError(f"Missing roles: {ROLES - set(role_counts)}")


def build(snapshot: Path = SNAPSHOT, source_manifest: Path = SOURCE_MANIFEST, dataset_dir: Path = DATASET) -> dict:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    installed = dataset_dir / "gold_v3_manifest.json"
    if installed.is_file():
        current = check_installed(dataset_dir)
        if current["snapshot_sha256"] == sha256(snapshot):
            return current
        raise ValueError("An installed GOLD dataset exists from a different snapshot")
    stage = Path(tempfile.mkdtemp(prefix=".gold_v3_stage_", dir=dataset_dir))
    handles = {}
    writers = {}
    try:
        for alias in CORPORA.values():
            handle = (stage / f"data_{alias}.csv").open("w", encoding="utf-8", newline="")
            handles[alias] = handle
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writers[alias] = writer
        docs: dict[str, set[str]] = defaultdict(set)
        unit_docs: dict[str, set[str]] = defaultdict(set)
        units: Counter[str] = Counter()
        roles: dict[str, Counter[str]] = defaultdict(Counter)
        for alias, row, accident_id in source_rows(snapshot, source_manifest):
            docs[alias].add(accident_id)
            if row is None:
                continue
            writers[alias].writerow(row)
            unit_docs[alias].add(accident_id)
            units[alias] += 1
            roles[alias][row["pred_label"]] += 1
        for handle in handles.values():
            handle.close()
        handles.clear()
        source_sha = sha256(snapshot)
        report = {
            "gold_version": "narrative_v3",
            "snapshot_sha256": source_sha,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_documents": sum(map(len, docs.values())),
            "total_units": sum(units.values()),
            "corpora": {},
        }
        for alias in CORPORA.values():
            file = stage / f"data_{alias}.csv"
            # Reparse staged data, including CSV quoting, before any replacement.
            seen_ids = set()
            seen_facts = set()
            with file.open(encoding="utf-8", newline="") as handle:
                rows = csv.DictReader(handle)
                if rows.fieldnames != FIELDS:
                    raise ValueError(f"Wrong schema: {alias}")
                count = 0
                for row in rows:
                    count += 1
                    if row["sentence"] != row["accident_summary"][int(row["start"]):int(row["end"])]:
                        raise ValueError(f"Staged excerpt mismatch: {alias}/{count}")
                    if row["doc_id"] in seen_ids or (row["accident_id"], row["fact_id"]) in seen_facts:
                        raise ValueError(f"Duplicate staged id: {alias}/{count}")
                    seen_ids.add(row["doc_id"])
                    seen_facts.add((row["accident_id"], row["fact_id"]))
                if count != units[alias] or len(seen_ids) != count:
                    raise ValueError(f"Staged count mismatch: {alias}")
            report["corpora"][alias] = {
                "source_corpus": next(key for key, value in CORPORA.items() if value == alias),
                "file": file.name,
                "sha256": sha256(file),
                "documents": len(docs[alias]),
                "documents_with_units": len(unit_docs[alias]),
                "documents_without_units": len(docs[alias] - unit_docs[alias]),
                "units": units[alias],
                "roles": dict(sorted(roles[alias].items())),
            }
        if report["total_documents"] != EXPECTED_DOCUMENTS or report["total_units"] != EXPECTED_UNITS:
            raise ValueError("Staged totals mismatch")

        existing = [dataset_dir / f"data_{alias}.csv" for alias in CORPORA.values()]
        old = [path for path in existing if path.exists()]
        archive = dataset_dir / "legacy_pre_gold_v3_20261018"
        archive.mkdir(exist_ok=True)
        for path in old:
            backup = archive / path.name
            if backup.exists() and sha256(backup) != sha256(path):
                raise ValueError(f"Existing legacy backup differs: {backup}")
            if not backup.exists():
                shutil.copy2(path, backup)
            if sha256(backup) != sha256(path):
                raise ValueError(f"Legacy backup verification failed: {path}")
        try:
            for alias in CORPORA.values():
                (stage / f"data_{alias}.csv").replace(dataset_dir / f"data_{alias}.csv")
            manifest_path = dataset_dir / "gold_v3_manifest.json"
            staged_manifest = stage / "gold_v3_manifest.json"
            staged_manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            staged_manifest.replace(manifest_path)
        except Exception:
            for path in old:
                shutil.copy2(archive / path.name, path)
            for path in existing:
                if path not in old:
                    path.unlink(missing_ok=True)
            raise
        return report
    finally:
        for handle in handles.values():
            handle.close()
        shutil.rmtree(stage, ignore_errors=True)


def check_installed(dataset_dir: Path = DATASET) -> dict:
    manifest = json.loads((dataset_dir / "gold_v3_manifest.json").read_text(encoding="utf-8"))
    if manifest["snapshot_sha256"] != EXPECTED_SHA or manifest["total_units"] != EXPECTED_UNITS:
        raise ValueError("Installed GOLD manifest does not match frozen RQ1")
    for alias, info in manifest["corpora"].items():
        file = dataset_dir / info["file"]
        if sha256(file) != info["sha256"]:
            raise ValueError(f"Installed dataset changed: {alias}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "check"])
    args = parser.parse_args()
    result = build() if args.command == "build" else check_installed()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
