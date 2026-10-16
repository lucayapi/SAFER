#!/usr/bin/env python3
"""Independent narrative-level preannotation via OpenAI Batch.

This is a *pilot* pipeline. Outputs are LLM proposals, never human gold.
Run from text/: python scripts/annotate_narratives_batch.py prepare --corpus bois_textille
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))
PROMPT_PATH = TEXT_ROOT / "annotation" / "prompts" / "narrative_units_v0.txt"
OUTPUT_ROOT = TEXT_ROOT / "output" / "narrative_annotation"
ROLES = ["A0", "A1", "B", "C", "AUTRE"]
ASSERTIONS = ["AFFIRME", "HYPOTHETIQUE", "NIE", "INCERTAIN"]


def _schema() -> dict[str, Any]:
    unit = {
        "type": "object",
        "properties": {
            "start": {"type": "integer"}, "end": {"type": "integer"},
            "text": {"type": "string"}, "role": {"type": "string", "enum": ROLES},
            "assertion": {"type": "string", "enum": ASSERTIONS},
            "uncertain": {"type": "boolean"}, "reason": {"type": "string"},
        },
        "required": ["start", "end", "text", "role", "assertion", "uncertain", "reason"],
        "additionalProperties": False,
    }
    excluded = {
        "type": "object",
        "properties": {
            "start": {"type": "integer"}, "end": {"type": "integer"},
            "reason": {"type": "string"},
        },
        "required": ["start", "end", "reason"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "document_id": {"type": "string"},
            "units": {"type": "array", "items": unit},
            "excluded_spans": {"type": "array", "items": excluded},
            "issues": {"type": "array", "items": {"type": "string"}},
            "needs_review": {"type": "boolean"},
        },
        "required": ["document_id", "units", "excluded_spans", "issues", "needs_review"],
        "additionalProperties": False,
    }


def _dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _slug(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Le nom de corpus doit contenir uniquement lettres, chiffres, _ ou -.")
    return value


def _client() -> Any:
    try:
        from dotenv import load_dotenv
        for path in (Path.cwd() / ".env", TEXT_ROOT / ".env", TEXT_ROOT / "scgm_text" / ".env"):
            if path.is_file():
                load_dotenv(path, override=False)
    except ImportError:
        pass
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY est absent de l'environnement ou de .env")
    from openai import OpenAI
    return OpenAI()


def _documents(csv_path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"accident_id", "fact_id", "sentence", "accident_summary"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Colonnes manquantes : {sorted(missing)}")
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in reader:
            grouped[row["accident_id"]].append(row)

    documents: list[dict[str, Any]] = []
    audit = {"narratives": 0, "sat_units": 0, "sat_uncovered_nonspace_spans": 0,
             "sat_uncovered_nonspace_chars": 0}
    for doc_id, rows in grouped.items():
        summaries = {row["accident_summary"] for row in rows}
        if len(summaries) != 1 or not next(iter(summaries)):
            raise ValueError(f"Récit absent ou contradictoire pour accident_id={doc_id}")
        if not doc_id or len({r["fact_id"] for r in rows}) != len(rows):
            raise ValueError(f"Identifiants manquants ou fact_id dupliqués : {doc_id}")
        raw = rows[0]["accident_summary"]
        cursor = 0
        sat: list[dict[str, Any]] = []
        for row in rows:
            sentence = row["sentence"]
            if not sentence:
                raise ValueError(f"Segment SaT vide : {doc_id}/{row['fact_id']}")
            start = raw.find(sentence, cursor)
            if start < 0:
                raise ValueError(f"Segment SaT introuvable à la suite : {doc_id}/{row['fact_id']}")
            gap = raw[cursor:start]
            if gap.strip():
                audit["sat_uncovered_nonspace_spans"] += 1
                audit["sat_uncovered_nonspace_chars"] += sum(not c.isspace() for c in gap)
            end = start + len(sentence)
            sat.append({"fact_id": row["fact_id"], "start": start, "end": end,
                        "text": sentence})
            cursor = end
        gap = raw[cursor:]
        if gap.strip():
            audit["sat_uncovered_nonspace_spans"] += 1
            audit["sat_uncovered_nonspace_chars"] += sum(not c.isspace() for c in gap)
        documents.append({"document_id": doc_id, "raw_text": raw,
                          "sat_segments": sat})
        audit["sat_units"] += len(sat)
    audit["narratives"] = len(documents)
    return documents, audit


def prepare(args: argparse.Namespace) -> None:
    corpus = _slug(args.corpus)
    source = Path(args.input_csv) if args.input_csv else (
        TEXT_ROOT / "annotation" / "data" / f"{corpus}_sentence_accidents.csv"
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    documents, audit = _documents(source)
    if not args.all:
        if args.limit <= 0:
            raise ValueError("--limit doit être positif")
        documents = random.Random(args.seed).sample(documents, min(args.limit, len(documents)))
    documents.sort(key=lambda d: d["document_id"])

    run_id = _slug(args.run_id) if args.run_id else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_ROOT / corpus / run_id
    if run_dir.exists():
        manifest_path = run_dir / "manifest.json"
        state = (json.loads(manifest_path.read_text(encoding="utf-8")).get("status")
                 if manifest_path.is_file() else "unknown")
        raise SystemExit(
            f"La campagne {run_dir} existe déjà (status={state}). "
            "Utilise son dossier avec submit/status/download/validate, "
            "ou choisis un nouveau --run-id pour préparer une autre campagne."
        )
    run_dir.mkdir(parents=True, exist_ok=False)
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    manifest = {
        "corpus": corpus, "run_id": run_id, "status": "prepared",
        "source_csv": str(source.resolve()), "source_sha256": _hash(source),
        "prompt_path": str(PROMPT_PATH), "prompt_sha256": _hash(PROMPT_PATH),
        "model": args.model, "reasoning_effort": args.reasoning_effort,
        "max_completion_tokens": args.max_completion_tokens,
        "sample_seed": None if args.all else args.seed,
        "n_requested": len(documents), "source_audit": audit,
        "annotation_status": "LLM_PREANNOTATION_ONLY",
    }
    with (run_dir / "documents.jsonl").open("w", encoding="utf-8") as docs_file, \
         (run_dir / "batch_input.jsonl").open("w", encoding="utf-8") as batch_file:
        for d in documents:
            docs_file.write(json.dumps(d, ensure_ascii=False) + "\n")
            request = {
                "custom_id": d["document_id"], "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content":
                         f"DOCUMENT_ID : {d['document_id']}\nRÉCIT ORIGINAL :\n{d['raw_text']}"},
                    ],
                    "response_format": {"type": "json_schema", "json_schema": {
                        "name": "narrative_units_v0", "strict": True, "schema": _schema()}},
                    "reasoning_effort": args.reasoning_effort,
                    "max_completion_tokens": args.max_completion_tokens,
                },
            }
            batch_file.write(json.dumps(request, ensure_ascii=False) + "\n")
    manifest["batch_input_sha256"] = _hash(run_dir / "batch_input.jsonl")
    _dump(run_dir / "manifest.json", manifest)
    print(json.dumps({"run_dir": str(run_dir), "n_requested": len(documents),
                      "batch_input_bytes": (run_dir / "batch_input.jsonl").stat().st_size,
                      "source_audit": audit}, ensure_ascii=False, indent=2))


def _manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def submit(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    manifest = _manifest(run_dir)
    if manifest["status"] != "prepared":
        raise RuntimeError(f"Soumission déjà effectuée ou état invalide : {manifest['status']}")
    if _hash(run_dir / "batch_input.jsonl") != manifest["batch_input_sha256"]:
        raise RuntimeError("batch_input.jsonl a changé depuis sa préparation")
    client = _client()
    with (run_dir / "batch_input.jsonl").open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    # Save the uploaded ID even if job creation fails; this prevents silent duplication.
    manifest["input_file_id"] = uploaded.id
    manifest["status"] = "uploaded"
    _dump(run_dir / "manifest.json", manifest)
    batch = client.batches.create(input_file_id=uploaded.id,
                                  endpoint="/v1/chat/completions",
                                  completion_window="24h")
    manifest["batch_id"] = batch.id
    manifest["status"] = batch.status
    _dump(run_dir / "manifest.json", manifest)
    print(json.dumps({"batch_id": batch.id, "status": batch.status,
                      "run_dir": str(run_dir)}, ensure_ascii=False, indent=2))


def status(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    manifest = _manifest(run_dir)
    if not manifest.get("batch_id"):
        raise RuntimeError("Aucun batch soumis pour ce dossier")
    batch = _client().batches.retrieve(manifest["batch_id"])
    manifest["status"] = batch.status
    manifest["request_counts"] = (batch.request_counts.model_dump()
                                   if batch.request_counts else None)
    manifest["output_file_id"] = batch.output_file_id
    manifest["error_file_id"] = batch.error_file_id
    _dump(run_dir / "manifest.json", manifest)
    print(json.dumps({"batch_id": batch.id, "status": batch.status,
                      "request_counts": manifest["request_counts"]},
                     ensure_ascii=False, indent=2))


def download(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    manifest = _manifest(run_dir)
    if not manifest.get("batch_id"):
        raise RuntimeError("Aucun batch soumis")
    client = _client()
    batch = client.batches.retrieve(manifest["batch_id"])
    if batch.status != "completed":
        raise RuntimeError(f"Le batch n'est pas terminé : {batch.status}")
    if not batch.output_file_id:
        raise RuntimeError("Batch terminé sans fichier de sortie")
    for file_id, name in ((batch.output_file_id, "batch_output.jsonl"),
                          (batch.error_file_id, "batch_errors.jsonl")):
        if file_id:
            dest = run_dir / name
            if dest.exists():
                raise FileExistsError(f"Fichier déjà téléchargé : {dest}")
            dest.write_bytes(client.files.content(file_id).content)
    manifest["status"] = "downloaded"
    manifest["output_file_id"] = batch.output_file_id
    manifest["error_file_id"] = batch.error_file_id
    _dump(run_dir / "manifest.json", manifest)
    print(run_dir / "batch_output.jsonl")


def validate_annotation(raw: str, annotation: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(annotation.get("units"), list) or not isinstance(annotation.get("excluded_spans"), list):
        return ["units ou excluded_spans absent ou invalide"]
    if not isinstance(annotation.get("issues"), list) or type(annotation.get("needs_review")) is not bool:
        errors.append("issues ou needs_review absent ou invalide")
    spans: list[tuple[int, int, str]] = []
    prev_end = 0
    for i, unit in enumerate(annotation.get("units", [])):
        start, end = unit.get("start"), unit.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(raw):
            errors.append(f"unit {i}: offsets invalides")
            continue
        if start < prev_end:
            errors.append(f"unit {i}: chevauchement ou ordre incorrect")
        if raw[start:end] != unit.get("text"):
            errors.append(f"unit {i}: texte différent du récit")
        if unit.get("role") not in ROLES or unit.get("assertion") not in ASSERTIONS:
            errors.append(f"unit {i}: rôle ou assertion invalide")
        if type(unit.get("uncertain")) is not bool or not isinstance(unit.get("reason"), str):
            errors.append(f"unit {i}: uncertain ou reason invalide")
        prev_end = end
        spans.append((start, end, "unit"))
    for i, span in enumerate(annotation.get("excluded_spans", [])):
        start, end = span.get("start"), span.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(raw):
            errors.append(f"excluded_span {i}: offsets invalides")
        else:
            spans.append((start, end, "excluded"))
        if not isinstance(span.get("reason"), str):
            errors.append(f"excluded_span {i}: reason invalide")
    spans.sort()
    for left, right in zip(spans, spans[1:]):
        if right[0] < left[1]:
            errors.append(f"chevauchement {left[2]}/{right[2]} à {right[0]}")
    return errors


def uncovered_alnum_chars(raw: str, annotation: dict[str, Any]) -> int:
    spans = sorted((s["start"], s["end"]) for key in ("units", "excluded_spans")
                   for s in annotation[key])
    cursor = 0
    uncovered = 0
    for start, end in spans:
        uncovered += sum(char.isalnum() for char in raw[cursor:start])
        cursor = end
    return uncovered + sum(char.isalnum() for char in raw[cursor:])


def validate(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    docs = {d["document_id"]: d for d in _read_jsonl(run_dir / "documents.jsonl")}
    responses = _read_jsonl(run_dir / "batch_output.jsonl")
    results: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    raw_needs_review = 0
    seen: set[str] = set()
    for response in responses:
        doc_id = str(response.get("custom_id"))
        errors: list[str] = []
        if doc_id not in docs:
            errors.append("document_id inconnu")
        if doc_id in seen:
            errors.append("réponse dupliquée")
        seen.add(doc_id)
        body = (response.get("response") or {}).get("body") or {}
        choices = body.get("choices") or []
        if not choices or choices[0].get("finish_reason") != "stop":
            errors.append("réponse absente ou tronquée")
        annotation = None
        if not errors:
            try:
                annotation = json.loads(choices[0]["message"]["content"])
            except (ValueError, KeyError, TypeError):
                errors.append("JSON absent ou illisible")
        if annotation is not None and not isinstance(annotation, dict):
            errors.append("Annotation JSON différente d'un objet")
            annotation = None
        if annotation is not None:
            raw_needs_review += annotation.get("needs_review") is True
            if annotation.get("document_id") != doc_id:
                errors.append("document_id de l'annotation différent")
            errors.extend(validate_annotation(docs[doc_id]["raw_text"], annotation))
        uncovered = (uncovered_alnum_chars(docs[doc_id]["raw_text"], annotation)
                     if annotation is not None and not errors else None)
        results.append({"document_id": doc_id, "valid": not errors, "errors": errors,
                        "needs_review": annotation.get("needs_review") if annotation else None,
                        "uncovered_alnum_chars": uncovered,
                        "review_recommended": bool(annotation and (
                            annotation.get("needs_review") or uncovered))})
        if annotation is not None and not errors:
            annotations.append({"annotation_source": "LLM", "review_status": "PENDING",
                                **annotation})
    missing = set(docs) - seen
    for doc_id in sorted(missing):
        results.append({"document_id": doc_id, "valid": False,
                        "errors": ["réponse manquante"], "needs_review": None})
    with (run_dir / "quality_report.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    with (run_dir / "llm_annotations.jsonl").open("w", encoding="utf-8") as handle:
        for annotation in annotations:
            handle.write(json.dumps(annotation, ensure_ascii=False) + "\n")
    summary = {"n_expected": len(docs), "n_responses": len(responses),
               "n_valid": len(annotations), "n_invalid_or_missing": len(docs) - len(annotations),
               "n_needs_review": sum(a["needs_review"] for a in annotations),
               "n_needs_review_all_raw_responses": raw_needs_review,
               "needs_review_denominator": "n_needs_review counts only technically valid annotations"}
    _dump(run_dir / "quality_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if len(annotations) != len(docs):
        raise SystemExit("Annotations incomplètes ou invalides : consulter quality_report.jsonl")


def _locate_units(raw: str, annotation: dict[str, Any]) -> list[dict[str, Any]]:
    """Locate literal quotes in narrative order; never guess among repetitions."""
    cursor = 0
    reports: list[dict[str, Any]] = []
    for index, unit in enumerate(annotation.get("units", [])):
        if not isinstance(unit, dict):
            reports.append({"unit_index": index, "status": "invalid_unit",
                            "proposed_start": None, "proposed_end": None})
            continue
        start, end, quote = unit.get("start"), unit.get("end"), unit.get("text")
        if not isinstance(quote, str) or not quote:
            state, proposed, candidate_count = "invalid_quote", None, 0
        else:
            candidates: list[int] = []
            at = raw.find(quote, cursor)
            while at >= 0:
                candidates.append(at)
                at = raw.find(quote, at + 1)
            candidate_count = len(candidates)
            if candidate_count == 1:
                proposed = candidates[0]
                state = ("original_exact" if type(start) is int and type(end) is int
                         and start == proposed and end == proposed + len(quote)
                         else "unique_exact_quote")
            elif candidate_count:
                state, proposed = "ambiguous_quote", None
            else:
                state, proposed = "quote_not_found", None
        proposed_end = proposed + len(quote) if proposed is not None else None
        if proposed_end is not None:
            cursor = proposed_end
        reports.append({"unit_index": index, "status": state,
                        "original_start": start, "original_end": end,
                        "proposed_start": proposed, "proposed_end": proposed_end,
                        "candidate_count": candidate_count})
    return reports


def reconcile_annotation(raw: str, annotation: dict[str, Any],
                         doc_id: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Create a traceable LLM draft only when every literal span is unambiguous."""
    errors: list[str] = []
    if annotation.get("document_id") != doc_id:
        errors.append("document_id de l'annotation différent")
    if not isinstance(annotation.get("units"), list) or not isinstance(annotation.get("excluded_spans"), list):
        errors.append("units ou excluded_spans absent ou invalide")
        reports: list[dict[str, Any]] = []
    else:
        reports = _locate_units(raw, annotation)
    if any(row["proposed_start"] is None for row in reports):
        errors.append("unité introuvable ou ambiguë : correction manuelle requise")
    draft = copy.deepcopy(annotation)
    for row in reports:
        if row["proposed_start"] is None:
            continue
        unit = draft["units"][row["unit_index"]]
        unit["offset_provenance"] = {
            "model_start": row["original_start"], "model_end": row["original_end"],
            "method": row["status"]}
        unit["start"], unit["end"] = row["proposed_start"], row["proposed_end"]
    if not errors:
        errors.extend(validate_annotation(raw, draft))
    if not errors:
        draft["annotation_source"] = "LLM"
        draft["review_status"] = "PENDING"
        draft["needs_review"] = True
        draft["reconciliation"] = {
            "method": "ordered_unique_exact_quote_v1",
            "n_relocated_units": sum(r["status"] == "unique_exact_quote" for r in reports),
            "excluded_offsets_verified_by_quote": False if draft["excluded_spans"] else None,
        }
    report = {"document_id": doc_id, "reconciled": not errors, "errors": errors,
              "unit_alignments": reports,
              "n_relocated_units": sum(r["status"] == "unique_exact_quote" for r in reports),
              "excluded_span_count": len(annotation.get("excluded_spans", []))
              if isinstance(annotation.get("excluded_spans"), list) else None,
              "uncovered_alnum_chars": uncovered_alnum_chars(raw, draft)
              if not errors else None,
              "requires_human_review": True}
    return (draft if not errors else None), report


def reconcile(args: argparse.Namespace) -> None:
    """Reuse downloaded responses locally; preserve their original contents."""
    run_dir = Path(args.run_dir).resolve()
    docs = {d["document_id"]: d["raw_text"]
            for d in _read_jsonl(run_dir / "documents.jsonl")}
    output_path = run_dir / "batch_output.jsonl"
    responses = _read_jsonl(output_path)
    manifest = _manifest(run_dir)
    output_hash = _hash(output_path)
    drafts: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    seen: set[str] = set()
    response_counts = Counter(str(r.get("custom_id")) for r in responses)
    for response in responses:
        doc_id = str(response.get("custom_id"))
        if doc_id in seen:
            continue
        seen.add(doc_id)
        if response_counts[doc_id] != 1 or doc_id not in docs:
            reports.append({"document_id": doc_id, "reconciled": False,
                            "errors": ["réponse dupliquée ou document_id inconnu"]})
            continue
        body = (response.get("response") or {}).get("body") or {}
        choices = body.get("choices") or []
        if not choices or choices[0].get("finish_reason") != "stop":
            reports.append({"document_id": doc_id, "reconciled": False,
                            "errors": ["réponse absente ou tronquée"]})
            continue
        try:
            annotation = json.loads(choices[0]["message"]["content"])
        except (ValueError, KeyError, TypeError):
            annotation = None
        if not isinstance(annotation, dict):
            reports.append({"document_id": doc_id, "reconciled": False,
                            "errors": ["JSON absent ou illisible"]})
            continue
        draft, report = reconcile_annotation(docs[doc_id], annotation, doc_id)
        reports.append(report)
        if draft is not None:
            draft["provenance"] = {
                "batch_id": manifest.get("batch_id"),
                "prompt_sha256": manifest.get("prompt_sha256"),
                "source_sha256": manifest.get("source_sha256"),
                "batch_output_sha256": output_hash,
            }
            drafts.append(draft)
    for doc_id in sorted(set(docs) - seen):
        reports.append({"document_id": doc_id, "reconciled": False,
                        "errors": ["réponse manquante"]})
    for name, rows in (("reconciled_annotations.jsonl", drafts),
                       ("reconciliation_report.jsonl", reports)):
        with (run_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "n_expected": len(docs), "n_responses": len(responses),
        "n_reconciled_drafts": len(drafts), "n_blocked": len(docs) - len(drafts),
        "n_relocated_units": sum(r.get("n_relocated_units", 0) for r in reports),
        "n_unlocatable_or_ambiguous_units": sum(
            u["proposed_start"] is None for r in reports
            for u in r.get("unit_alignments", [])),
        "annotation_status": "LLM_PREANNOTATION_ONLY",
        "note": "Alignement textuel automatique uniquement ; rôles, faits et exclusions doivent être relus humainement.",
    }
    _dump(run_dir / "reconciliation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def diagnose_offsets(args: argparse.Namespace) -> None:
    """Suggest exact-quote offsets without modifying or approving LLM responses."""
    run_dir = Path(args.run_dir).resolve()
    docs = {d["document_id"]: d["raw_text"]
            for d in _read_jsonl(run_dir / "documents.jsonl")}
    responses = _read_jsonl(run_dir / "batch_output.jsonl")
    diagnostics: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    for response in responses:
        doc_id = str(response.get("custom_id"))
        if doc_id not in docs:
            raise ValueError(f"Réponse inconnue : {doc_id}")
        body = (response.get("response") or {}).get("body") or {}
        choices = body.get("choices") or []
        if not choices or choices[0].get("finish_reason") != "stop":
            raise ValueError(f"Réponse absente ou tronquée : {doc_id}")
        annotation = json.loads(choices[0]["message"]["content"])
        raw = docs[doc_id]
        cursor = 0
        unit_reports: list[dict[str, Any]] = []
        located_spans: list[tuple[int, int]] = []
        for index, unit in enumerate(annotation["units"]):
            start, end, quote = unit["start"], unit["end"], unit["text"]
            if not isinstance(quote, str) or not quote:
                raise ValueError(f"Texte d'unité vide ou invalide : {doc_id}/{index}")
            if type(start) is int and type(end) is int and 0 <= start < end <= len(raw) \
                    and raw[start:end] == quote \
                    and start >= cursor:
                state, proposed = "original_exact", start
                candidate_count = 1
            else:
                candidates: list[int] = []
                at = raw.find(quote, cursor)
                while at >= 0:
                    candidates.append(at)
                    at = raw.find(quote, at + 1)
                candidate_count = len(candidates)
                if candidate_count == 1:
                    state, proposed = "unique_exact_quote", candidates[0]
                elif candidate_count > 1:
                    state, proposed = "ambiguous_quote", None
                else:
                    state, proposed = "quote_not_found", None
            counts[state] += 1
            if proposed is not None:
                cursor = proposed + len(quote)
                located_spans.append((proposed, cursor))
            unit_reports.append({
                "unit_index": index, "status": state,
                "original_start": start, "original_end": end,
                "proposed_start": proposed,
                "proposed_end": proposed + len(quote) if proposed is not None else None,
                "candidate_count": candidate_count,
            })
        excluded_overlap = 0
        for span in annotation["excluded_spans"]:
            s, e = span["start"], span["end"]
            if type(s) is int and type(e) is int and any(s < ue and us < e
                                                         for us, ue in located_spans):
                excluded_overlap += 1
        if excluded_overlap:
            counts["excluded_spans_overlapping_located_units"] += excluded_overlap
        diagnostics.append({
            "document_id": doc_id, "units": unit_reports,
            "all_unit_quotes_located": all(r["proposed_start"] is not None
                                           for r in unit_reports),
            "excluded_spans_overlapping_located_units": excluded_overlap,
            "requires_human_review": True,
        })
    with (run_dir / "offset_diagnostics.jsonl").open("w", encoding="utf-8") as handle:
        for row in diagnostics:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"n_responses": len(responses), "unit_status_counts": dict(counts),
               "n_documents_all_unit_quotes_located": sum(
                   row["all_unit_quotes_located"] for row in diagnostics),
               "n_documents_without_excluded_overlap": sum(
                   row["excluded_spans_overlapping_located_units"] == 0
                   for row in diagnostics),
               "note": "Positions proposées à partir des extraits exacts ; aucune correction automatique ni validation sémantique."}
    _dump(run_dir / "offset_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Préparer localement sans appel à l'API")
    prep.add_argument("--corpus", required=True, help="Ex. bois_textille")
    prep.add_argument("--input-csv", help="CSV source personnalisé")
    prep.add_argument("--run-id", help="Identifiant de campagne personnalisé")
    sample = prep.add_mutually_exclusive_group()
    sample.add_argument("--limit", type=int, default=20, help="Pilote de 20 récits par défaut")
    sample.add_argument("--all", action="store_true", help="Tous les récits (explicite)")
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--model", default="gpt-5.4-mini")
    prep.add_argument("--reasoning-effort", choices=["none", "low", "medium"], default="low")
    prep.add_argument("--max-completion-tokens", type=int, default=12000)
    prep.set_defaults(func=prepare)
    for name, fn in (("submit", submit), ("status", status),
                     ("download", download), ("validate", validate),
                     ("diagnose-offsets", diagnose_offsets),
                     ("reconcile", reconcile)):
        command = sub.add_parser(name)
        command.add_argument("--run-dir", required=True)
        command.set_defaults(func=fn)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
