"""Create a derived GOLD dataset without explicitly excluded documentary spans.

The frozen RQ1 snapshot and the existing GOLD datasets are never modified.
Factual units are retained verbatim; only their offsets in the cleaned raw
narrative are recomputed. The output is suitable for experiments whose input
is defined as the accident narrative rather than the full documentary report.
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

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "annotation" / "rq1_sat_v3_20261017" / "gold_snapshot.jsonl"
SOURCE_MANIFEST = ROOT / "annotation" / "rq1_sat_v3_20261017" / "manifest.json"
DEFAULT_OUTPUT = ROOT / "text" / "dataset" / "narrative_only_v1"
EXPECTED_SHA = "4845dbec927c5cd1161360729382ac5497c3f5c41d1562b03d5020c2f20bdca8"
CORPORA = {
    "btp": "btp", "metallurgie": "metallurgie",
    "caou_chimie_plas": "caou", "bois_textille": "bois_textille",
    "service_temporaire": "service_temporaire", "nicollin": "nicollin",
}
CSV_FIELDS = [
    "accident_id", "fact_id", "sentence", "accident_summary", "pred_label",
    "pred_ok", "doc_id", "start", "end", "uncertain", "validation_method",
    "gold_version",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_text(raw: str, exclusions: list[dict], units: list[dict]):
    spans = sorted((int(s["start"]), int(s["end"])) for s in exclusions)
    last = 0
    for start, end in spans:
        if not (last <= start < end <= len(raw)):
            raise ValueError("Invalid or overlapping excluded span")
        last = end
    for unit in units:
        a, b = int(unit["start"]), int(unit["end"])
        if not (0 <= a < b <= len(raw)):
            raise ValueError("Invalid factual-unit offsets")
        if any(a < e and s < b for s, e in spans):
            raise ValueError("A factual unit overlaps an explicitly excluded span")

    # Map each retained original character to its new offset. Replace each
    # removed passage by one separator only when deletion would join two
    # non-whitespace characters. All factual substrings remain byte-for-byte
    # (Unicode code-point-for-code-point) identical.
    chars: list[str] = []
    original_to_clean: dict[int, int] = {}
    cursor = 0
    for start, end in spans:
        chars.extend(raw[cursor:start])
        for i in range(cursor, start):
            original_to_clean[i] = len(chars) - (start - i)
        left = raw[start - 1] if start > 0 else ""
        right = raw[end] if end < len(raw) else ""
        if left and right and not left.isspace() and not right.isspace():
            chars.append(" ")
        cursor = end
    tail_start = cursor
    chars.extend(raw[tail_start:])
    for i in range(tail_start, len(raw)):
        original_to_clean[i] = len(chars) - (len(raw) - i)
    cleaned = "".join(chars)

    remapped = []
    for unit in units:
        a, b = int(unit["start"]), int(unit["end"])
        new_a = original_to_clean[a]
        new_b = original_to_clean[b - 1] + 1
        if cleaned[new_a:new_b] != raw[a:b]:
            raise ValueError("Offset remapping changed a factual unit")
        remapped.append({**unit, "start": new_a, "end": new_b})
    return cleaned, remapped


def build(output: Path) -> dict:
    actual_sha = sha256(SOURCE)
    if actual_sha != EXPECTED_SHA:
        raise ValueError(f"Unexpected source snapshot SHA256: {actual_sha}")
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if source_manifest["signature"]["snapshot_sha256"] != actual_sha:
        raise ValueError("Source manifest does not match the frozen snapshot")
    if output.exists():
        manifest_file = output / "narrative_only_manifest.json"
        if manifest_file.exists():
            old = json.loads(manifest_file.read_text(encoding="utf-8"))
            if old.get("source_snapshot_sha256") == actual_sha and old.get("version") == "narrative_only_v1":
                return old
        raise FileExistsError(f"Output exists and is not a verified build: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".narrative_only_stage_", dir=output.parent))
    facts_by_corpus: dict[str, list[dict]] = defaultdict(list)
    docs_by_corpus: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    unit_counts: Counter[str] = Counter()
    roles_by_corpus: dict[str, Counter] = defaultdict(Counter)
    snapshot_path = stage / "gold_snapshot.jsonl"
    try:
        with SOURCE.open(encoding="utf-8") as src, snapshot_path.open("w", encoding="utf-8", newline="\n") as dst:
            for line_no, line in enumerate(src, 1):
                row = json.loads(line)
                corpus = CORPORA.get(row["corpus"])
                if corpus is None:
                    raise ValueError(f"Unknown corpus at line {line_no}")
                raw = row["raw_text"]
                units = row.get("units", [])
                exclusions = row.get("excluded_spans", [])
                cleaned, new_units = clean_text(raw, exclusions, units)
                new_row = {**row, "raw_text": cleaned, "units": new_units,
                           "excluded_spans": []}
                new_row["preprocessing"] = "explicit_excluded_spans_removed_v1"
                dst.write(json.dumps(new_row, ensure_ascii=False) + "\n")
                docs_by_corpus[corpus] += 1
                exclusion_counts[corpus] += len(exclusions)
                for fact_id, unit in enumerate(new_units, 1):
                    text = cleaned[unit["start"]:unit["end"]]
                    facts_by_corpus[corpus].append({
                        "accident_id": row["document_id"], "fact_id": fact_id,
                        "sentence": text, "accident_summary": cleaned,
                        "pred_label": unit["role"], "pred_ok": True,
                        "doc_id": len(facts_by_corpus[corpus]) + 1,
                        "start": unit["start"], "end": unit["end"],
                        "uncertain": bool(unit.get("uncertain", False)),
                        "validation_method": row.get("validation_method", ""),
                        "gold_version": "narrative_v3_narrative_only_v1",
                    })
                    unit_counts[corpus] += 1
                    roles_by_corpus[corpus][unit["role"]] += 1

        corpus_manifest = {}
        for corpus in CORPORA.values():
            csv_path = stage / f"data_{corpus}.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(facts_by_corpus[corpus])
            # Verify every exported fact is an exact substring of its cleaned narrative.
            with csv_path.open(encoding="utf-8", newline="") as f:
                for ix, exported in enumerate(csv.DictReader(f), 1):
                    a, b = int(exported["start"]), int(exported["end"])
                    if exported["sentence"] != exported["accident_summary"][a:b]:
                        raise ValueError(f"CSV span mismatch: {corpus}/{ix}")
            corpus_manifest[corpus] = {
                "file": csv_path.name, "documents": docs_by_corpus[corpus],
                "units": unit_counts[corpus], "removed_documentary_spans": exclusion_counts[corpus],
                "roles": dict(sorted(roles_by_corpus[corpus].items())),
                "csv_sha256": sha256(csv_path),
            }
        manifest = {
            "version": "narrative_only_v1",
            "source_snapshot_sha256": actual_sha,
            "source_manifest_sha256": sha256(SOURCE_MANIFEST),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "operation": "Remove only spans explicitly listed in excluded_spans; retain factual units verbatim and remap offsets.",
            "total_documents": sum(docs_by_corpus.values()),
            "total_units": sum(unit_counts.values()),
            "total_removed_documentary_spans": sum(exclusion_counts.values()),
            "corpora": corpus_manifest,
        }
        (stage / "narrative_only_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        readme = (
            "# Récits nettoyés — narrative_only_v1\n\n"
            "Dérivé du snapshot GOLD RQ1 `narrative_v3`. Chaque span explicitement présent dans `excluded_spans` a été retiré du texte brut; les unités factuelles sont conservées à l’identique et leurs offsets sont recalculés. Les originaux ne sont pas modifiés. Ce nettoyage n’infère aucune autre suppression.\n\n"
            "`gold_snapshot.jsonl` contient tous les récits, y compris ceux sans unité. `data_<corpus>.csv` contient les unités factuelles, leur texte exact et le récit nettoyé. Les hashes et effectifs sont dans `narrative_only_manifest.json`.\n\n"
            "Ce jeu convient aux expériences où l’entrée est le récit débarrassé des passages documentaires explicitement annotés. Il ne faut pas le présenter comme une annotation indépendante : les spans retirés proviennent des exclusions du GOLD existant.\n"
        )
        (stage / "README.md").write_text(readme, encoding="utf-8")
        stage.replace(output)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
