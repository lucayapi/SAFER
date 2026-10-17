#!/usr/bin/env python3
"""Prepare and run six exploratory SaT-LoRA leave-one-corpus-out experiments.

No narratives are committed: prepare writes only to an explicit, ignored bundle.
The comparison reference is AI preannotation, never human-validated gold.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "text" / "annotation" / "sat_lodo_lora_template.json"
CORPORA = ("btp", "metallurgie", "caou_chimie_plas", "bois_textille",
           "service_temporaire", "nicollin")
EXPECTED = dict(zip(CORPORA, (6040, 3212, 838, 1920, 2939, 5673)))
SOURCES = {"CODEX_DIRECT_PREANNOTATION", "OPENAI_BATCH_PREANNOTATION"}
ROLES = {"A0", "A1", "B", "C", "AUTRE"}
ASSERTIONS = {"AFFIRME", "HYPOTHETIQUE", "NIE", "INCERTAIN"}
MODEL = "sat-12l-sm"
VENDOR_REVISION = "3ef928c8e163d11dc6f6336ad610bff2ffe0bda4"
DEFAULT = 0.25
THRESHOLDS = (0.01, 0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
SEED = 20261017
DATASET_NAME = "safer_lodo"


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_record(raw: str, ann: dict) -> str | None:
    """Return the first exclusion reason; the model learns only unambiguous facts."""
    if ann.get("annotation_source") not in SOURCES:
        return "not_direct"
    if ann.get("guide_version") != "narrative_v2":
        return "wrong_guide"
    if ann.get("issues"):
        return "issues"
    units = ann.get("units")
    excluded = ann.get("excluded_spans")
    if not isinstance(units, list) or not units or not isinstance(excluded, list):
        return "missing_units"
    if any(not isinstance(u, dict) or u.get("uncertain") is not False for u in units):
        return "uncertain_unit"
    spans = []
    for kind, entries in (("unit", units), ("excluded", excluded)):
        for item in entries:
            if not isinstance(item, dict):
                return "invalid_span"
            start, end = item.get("start"), item.get("end")
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(raw):
                return "invalid_span"
            if raw[start:end] != item.get("text"):
                return "inexact_text"
            if kind == "unit" and (item.get("role") not in ROLES or
                                    item.get("assertion") not in ASSERTIONS):
                return "invalid_label"
            spans.append((start, end, kind))
    spans.sort()
    if any(b[0] < a[1] for a, b in zip(spans, spans[1:])):
        return "overlap"
    cursor = 0
    for unit in units:
        if unit["start"] < cursor:
            return "unordered_units"
        if "\n" in unit["text"] or "\r" in unit["text"]:
            return "newline_in_unit"
        if any(c.isalnum() for c in raw[cursor:unit["start"]]):
            return "informative_gap"
        cursor = unit["end"]
    if any(c.isalnum() for c in raw[cursor:]):
        return "informative_gap"
    return None


def training_chunks(raw: str, units: list[dict]) -> tuple[list[str], int]:
    """Put noninformative gaps before the next unit, keeping factual end cuts.

    Only the derived training copy replaces line breaks in gaps by spaces.
    The untouched source text and offsets remain in fold JSONL files.
    """
    chunks = []
    cursor = 0
    changed = 0
    for unit in units:
        gap = raw[cursor:unit["start"]]
        changed += gap.count("\n") + gap.count("\r")
        gap = gap.replace("\r", " ").replace("\n", " ")
        chunks.append(gap + unit["text"])
        cursor = unit["end"]
    tail = raw[cursor:]
    changed += tail.count("\n") + tail.count("\r")
    chunks[-1] += tail.replace("\r", " ").replace("\n", " ")
    if any(not chunk or "\n" in chunk or "\r" in chunk for chunk in chunks):
        raise ValueError("Invalid derived training chunk")
    if "".join(chunks) != raw.replace("\r", " ").replace("\n", " "):
        raise AssertionError("Derived training text does not reconstruct source")
    return chunks, changed


def load_corpora(annotation_root: Path, *, check_counts: bool = True) -> tuple[dict, dict]:
    records: dict[str, dict[str, dict]] = {}
    report: dict[str, dict] = {}
    seen_ids: set[str] = set()
    for corpus in CORPORA:
        documents_path = annotation_root / "data" / corpus / "documents.jsonl"
        if not documents_path.is_file():
            raise FileNotFoundError(documents_path)
        docs = {}
        for doc in read_jsonl(documents_path):
            ident, raw = doc.get("document_id"), doc.get("raw_text")
            if not isinstance(ident, str) or not isinstance(raw, str) or not raw or ident in docs:
                raise ValueError(f"Invalid or duplicate document in {documents_path}: {ident}")
            docs[ident] = raw
        if check_counts and len(docs) != EXPECTED[corpus]:
            raise ValueError(f"{corpus}: expected {EXPECTED[corpus]} documents, found {len(docs)}")
        duplicate = seen_ids.intersection(docs)
        if duplicate:
            raise ValueError(f"Cross-corpus accident IDs: {corpus}: {sorted(duplicate)[:3]}")
        seen_ids.update(docs)
        accepted = {}
        reasons = collections.Counter()
        ann_digest = hashlib.sha256()
        changed_newlines = 0
        for ident, raw in docs.items():
            path = annotation_root / "preannotations" / corpus / f"{ident}.json"
            if not path.is_file():
                reasons["missing_annotation"] += 1
                continue
            content = path.read_bytes()
            ann_digest.update(path.name.encode() + b"\0" + content + b"\0")
            try:
                ann = json.loads(content)
            except (ValueError, UnicodeDecodeError):
                reasons["invalid_json"] += 1
                continue
            if ann.get("document_id") != ident:
                reasons["wrong_id"] += 1
                continue
            reason = validate_record(raw, ann)
            if reason:
                reasons[reason] += 1
                continue
            chunks, n = training_chunks(raw, ann["units"])
            changed_newlines += n
            accepted[ident] = {"corpus": corpus, "document_id": ident,
                               "raw_text": raw, "units": [
                                   {k: u[k] for k in ("start", "end", "text", "role", "assertion")}
                                   for u in ann["units"]],
                               "training_chunks": chunks,
                               "annotation_source": ann["annotation_source"]}
        records[corpus] = accepted
        report[corpus] = {"documents": len(docs), "eligible": len(accepted),
                          "excluded": dict(sorted(reasons.items())),
                          "training_newline_characters_replaced": changed_newlines,
                          "documents_sha256": digest(documents_path),
                          "annotations_sha256": ann_digest.hexdigest()}
    return records, report


def partition(ids: list[str], corpus: str, seed: int) -> tuple[list[str], list[str]]:
    ordered = sorted(ids, key=lambda ident: (hashlib.sha256(
        f"{seed}:{corpus}:{ident}".encode()).digest(), ident))
    n_train = math.floor(0.8 * len(ordered))
    if not 0 < n_train < len(ordered):
        raise ValueError(f"No valid 80/20 split for {corpus}: {len(ids)} eligible")
    return ordered[:n_train], ordered[n_train:]


def preflight(args: argparse.Namespace) -> tuple[dict, dict]:
    data, report = load_corpora(Path(args.annotation_root).resolve(),
                                check_counts=not args.allow_fixture_counts)
    if getattr(args, "report", None):
        write_json(Path(args.report).resolve(), {"checked_at_utc": stamp(),
                                                  "source_report": report})
    print(json.dumps(report, ensure_ascii=False, indent=2))
    missing = [corpus for corpus in CORPORA if len(data[corpus]) < 2]
    if missing:
        raise RuntimeError("Cannot prepare six LODO folds; no train/validation split for: " +
                           ", ".join(missing))
    return data, report


def prepare(args: argparse.Namespace) -> None:
    data, report = preflight(args)
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    out.mkdir(parents=True)
    import torch

    split = {corpus: dict(zip(("train", "val"), partition(list(data[corpus]), corpus, args.seed)))
             for corpus in CORPORA}
    files = {}
    for target in CORPORA:
        folder = out / "folds" / f"target_{target}"
        folder.mkdir(parents=True)
        source_corpora = [c for c in CORPORA if c != target]
        subsets = {
            "train": [data[c][ident] for c in source_corpora for ident in split[c]["train"]],
            "val": [data[c][ident] for c in source_corpora for ident in split[c]["val"]],
            "test": [data[target][ident] for ident in sorted(data[target])],
        }
        for name, rows in subsets.items():
            path = folder / f"{name}.jsonl"
            write_jsonl(path, rows)
            files[str(path.relative_to(out)).replace("\\", "/")] = digest(path)
        ids = {name: {c: (split[c][name] if name != "test" else [])
                       for c in source_corpora} if name != "test" else {target: sorted(data[target])}
               for name in ("train", "val", "test")}
        write_json(folder / "split_ids.json", ids)
        files[str((folder / "split_ids.json").relative_to(out)).replace("\\", "/")] = digest(folder / "split_ids.json")
        dataset = {"fr": {"sentence": {DATASET_NAME: {
            "meta": {"train_data": [row["training_chunks"] for row in subsets["train"]]},
            "data": [row["training_chunks"] for row in subsets["val"]]}}}}
        pth = folder / "training.pth"
        torch.save(dataset, pth)
        files[str(pth.relative_to(out)).replace("\\", "/")] = digest(pth)
    manifest = {"created_at_utc": stamp(), "reference": "AI_DIRECT_PREANNOTATIONS_NOT_HUMAN_GOLD",
                "model": MODEL, "seed": args.seed, "corpora": list(CORPORA),
                "fold_order": list(CORPORA), "train_fraction": 0.8,
                "thresholds": list(THRESHOLDS), "default_threshold": DEFAULT,
                "eligibility": "direct v2; no issues/uncertain; exact nonoverlapping offsets; no alphanumeric gaps; no newlines inside units",
                "training_gap_policy": "gaps prefix next unit; trailing gap suffix last; CR/LF in gaps become spaces only in derived training chunks",
                "source_report": report, "template_sha256": digest(TEMPLATE), "files": files}
    write_json(out / "manifest.json", manifest)
    write_json(out / "complete.json", {"created_at_utc": stamp(), "folds": len(CORPORA)})
    print(json.dumps({"bundle": str(out), "folds": len(CORPORA),
                      "eligible": {c: len(data[c]) for c in CORPORA}}, ensure_ascii=False, indent=2))


def verify_bundle(bundle: Path) -> dict:
    manifest_path = bundle / "manifest.json"
    if not (bundle / "complete.json").is_file() or not manifest_path.is_file():
        raise RuntimeError("Incomplete data bundle")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("corpora") != list(CORPORA) or manifest.get("model") != MODEL:
        raise ValueError("Unknown bundle version or model")
    if manifest.get("template_sha256") != digest(TEMPLATE):
        raise ValueError("The LoRA template differs from the bundle")
    for relative, expected in manifest["files"].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle.resolve()) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Missing or modified bundle file: {relative}")
    for target in CORPORA:
        folder = bundle / "folds" / f"target_{target}"
        groups = {name: read_jsonl(folder / f"{name}.jsonl") for name in ("train", "val", "test")}
        keys = {name: {(r["corpus"], r["document_id"]) for r in rows}
                for name, rows in groups.items()}
        if any(len(keys[name]) != len(groups[name]) or not groups[name] for name in groups):
            raise ValueError(f"Empty or duplicate partition: {target}")
        if any(keys[a] & keys[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError(f"Partition overlap: {target}")
        if any(r["corpus"] == target for r in groups["train"] + groups["val"]):
            raise ValueError(f"Target leakage into sources: {target}")
        if any(r["corpus"] != target for r in groups["test"]):
            raise ValueError(f"Non-target test record: {target}")
    return manifest


def check_bundle(args: argparse.Namespace) -> None:
    manifest = verify_bundle(Path(args.bundle).resolve())
    print(json.dumps({"ready": True, "model": manifest["model"],
                      "folds": len(manifest["fold_order"]),
                      "eligible": {c: manifest["source_report"][c]["eligible"] for c in CORPORA}},
                     ensure_ascii=False, indent=2))


def fold_for_index(manifest: dict, index: int) -> str:
    if not 0 <= index < len(CORPORA):
        raise ValueError(f"task-index must be 0..{len(CORPORA)-1}")
    return manifest["fold_order"][index]


def official_trainer(repo: Path) -> Path:
    script = repo / "wtpsplit" / "train" / "train_lora.py"
    if not script.is_file():
        raise FileNotFoundError(f"Official trainer not found: {script}")
    return script


def git_revision(repo: Path) -> str:
    """Verify the vendored upstream snapshot before loading or training."""
    manifest_path = repo / "SAFER_VENDOR.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Vendored wtpsplit manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("files")
    revision = manifest.get("revision")
    if not isinstance(expected, dict) or revision != VENDOR_REVISION:
        raise ValueError("Invalid vendored wtpsplit manifest")
    hash_mode = manifest.get("hash_mode", "raw")
    binary_files = manifest.get("binary_files", [])
    if hash_mode not in {"raw", "lf_normalized_text"} or not isinstance(binary_files, list) or \
       not all(isinstance(item, str) for item in binary_files) or \
       not set(binary_files).issubset(expected):
        raise ValueError("Invalid vendored wtpsplit hash mode")
    actual = {p.relative_to(repo).as_posix() for p in repo.rglob("*")
              if p.is_file() and p != manifest_path and not p.name.endswith(".pyc")
              and not any(part in {"__pycache__", ".pytest_cache", "build", "dist"}
                          or part.endswith(".egg-info") for part in p.relative_to(repo).parts)}
    if actual != set(expected):
        raise RuntimeError("Vendored wtpsplit files differ from their manifest")
    for relative, recorded_hash in expected.items():
        content = (repo / relative).read_bytes()
        if hash_mode == "lf_normalized_text" and relative not in binary_files:
            content = content.replace(b"\r\n", b"\n")
        if hashlib.sha256(content).hexdigest() != recorded_hash:
            raise RuntimeError(f"Vendored wtpsplit file changed: {relative}")
    return revision


def gpu_runtime(repo: Path) -> dict:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("The LoRA smoke/fit command requires a CUDA GPU")
    major_minor = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if major_minor >= (2, 6):
        raise RuntimeError("Official train_lora.py uses torch.load without weights_only=False; use torch 2.5.x")
    versions = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    for package in ("wtpsplit", "transformers", "adapters", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Missing LoRA dependency: {package}") from exc
    spec = importlib.util.find_spec("wtpsplit")
    expected_package = (repo / "wtpsplit").resolve()
    if not spec or not spec.origin or Path(spec.origin).resolve().parent != expected_package:
        raise RuntimeError("Install the vendored wtpsplit checkout in SAT_LODO_VENV with pip install -e")
    from transformers.utils import is_safetensors_available
    if not is_safetensors_available():
        raise RuntimeError("Transformers cannot load sat-12l-sm model.safetensors; install safetensors in SAT_LODO_VENV")
    versions["wtpsplit_source"] = str(expected_package)
    return versions


def cached_hub_revision(model: str) -> str | None:
    from huggingface_hub.constants import HF_HUB_CACHE

    path = Path(HF_HUB_CACHE) / ("models--" + model.replace("/", "--")) / "refs" / "main"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else None


def adapter_folder(output_dir: Path) -> Path:
    return output_dir / DATASET_NAME / "fr"


def adapter_exists(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and any(
        (path / filename).is_file() for filename in
        ("pytorch_adapter.bin", "adapter_model.safetensors"))


def run_trainer(repo: Path, config: dict, config_path: Path) -> None:
    write_json(config_path, config)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["WANDB_DISABLED"] = "true"
    env["TOKENIZERS_PARALLELISM"] = "false"
    subprocess.run([sys.executable, "-u", str(official_trainer(repo)), str(config_path)],
                   cwd=repo, env=env, check=True)


def export_loss_history(trainer_state: Path, output_csv: Path) -> None:
    """Keep loss values independently of rotating training checkpoints."""
    state = json.loads(trainer_state.read_text(encoding="utf-8"))
    history = state.get("log_history")
    if not isinstance(history, list):
        raise ValueError(f"Missing log_history in {trainer_state}")
    rows = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        for metric, value in entry.items():
            if metric in {"loss", "eval_loss", "train_loss"} or metric.endswith("/loss"):
                rows.append({"step": entry.get("step"), "epoch": entry.get("epoch"),
                             "metric": metric, "value": value,
                             "learning_rate": entry.get("learning_rate")})
    write_csv(output_csv, rows, ("step", "epoch", "metric", "value", "learning_rate"))


def base_config(bundle: Path, target: str, output_dir: Path) -> dict:
    config = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    if (config.get("model_name_or_path") != "segment-any-text/" + MODEL or
            config.get("subsample") is not None or config.get("use_auxiliary") is not False or
            config.get("do_auxiliary_training") is not False or config.get("clf_from_scratch", False)):
        raise ValueError("Unexpected LoRA model, labels, or subsample in template")
    config.update({"text_path": str((bundle / "folds" / f"target_{target}" / "training.pth").resolve()),
                   "output_dir": str(output_dir.resolve()), "seed": SEED})
    return config


def load_sat(adapter: Path | None = None):
    from wtpsplit import SaT
    sat = SaT(MODEL, lora_path=str(adapter) if adapter else None)
    try:
        import torch
        if torch.cuda.is_available():
            sat.to("cuda")
    except ImportError:
        pass
    return sat


def smoke(args: argparse.Namespace) -> None:
    bundle = Path(args.bundle).resolve()
    manifest = verify_bundle(bundle)
    target = fold_for_index(manifest, args.task_index)
    repo = Path(args.wtpsplit_repo).resolve()
    revision = git_revision(repo)
    runtime = gpu_runtime(repo)
    root = Path(args.run_root).resolve()
    smoke_dir = root / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    marker = root / "smoke_ok.json"
    import torch
    original = torch.load(bundle / "folds" / f"target_{target}" / "training.pth",
                          map_location="cpu", weights_only=False)
    source = original["fr"]["sentence"][DATASET_NAME]
    subset = {"fr": {"sentence": {DATASET_NAME: {
        "meta": {"train_data": source["meta"]["train_data"][:8]},
        "data": source["data"][:4]}}}}
    if len(subset["fr"]["sentence"][DATASET_NAME]["data"]) < 2:
        raise ValueError("Not enough validation examples for smoke test")
    pth = smoke_dir / "training.pth"
    torch.save(subset, pth)
    config = base_config(bundle, target, smoke_dir / "trainer")
    config["seed"] = manifest["seed"]
    config.update({"text_path": str(pth), "num_train_epochs": 1,
                   "max_steps": 1, "save_steps": 1,
                   "per_device_train_batch_size": 2,
                   "per_device_eval_batch_size": 2,
                   "gradient_accumulation_steps": 1})
    run_trainer(repo, config, smoke_dir / "config.json")
    export_loss_history(smoke_dir / "trainer" / "trainer_state.json",
                        smoke_dir / "loss_history.csv")
    adapter = adapter_folder(smoke_dir / "trainer")
    if not adapter_exists(adapter):
        raise RuntimeError(f"Smoke adapter not saved at {adapter}")
    sat = load_sat(adapter)
    sample = read_jsonl(bundle / "folds" / f"target_{target}" / "val.jsonl")[0]["raw_text"]
    pieces = list(sat.split([sample], threshold=DEFAULT))[0]
    pieces_to_spans(sample, pieces)
    runtime["base_model_revision"] = cached_hub_revision("segment-any-text/" + MODEL)
    runtime["tokenizer_revision"] = cached_hub_revision("FacebookAI/xlm-roberta-base")
    write_json(marker, {"bundle_manifest_sha256": digest(bundle / "manifest.json"),
                        "template_sha256": digest(TEMPLATE),
                        "wtpsplit_revision": revision, "model": MODEL,
                        "adapter": str(adapter), "runtime": runtime,
                        "checked_at_utc": stamp()})
    print(json.dumps({"smoke_ok": True, "marker": str(marker), "adapter": str(adapter)}, indent=2))


def fit(args: argparse.Namespace) -> None:
    bundle = Path(args.bundle).resolve()
    manifest = verify_bundle(bundle)
    target = fold_for_index(manifest, args.task_index)
    repo = Path(args.wtpsplit_repo).resolve()
    revision = git_revision(repo)
    runtime = gpu_runtime(repo)
    root = Path(args.run_root).resolve()
    marker = root / "smoke_ok.json"
    if not marker.is_file():
        raise RuntimeError("Run the GPU smoke test before the six full tasks")
    passed = json.loads(marker.read_text(encoding="utf-8"))
    if passed.get("bundle_manifest_sha256") != digest(bundle / "manifest.json") or \
       passed.get("template_sha256") != digest(TEMPLATE) or \
       passed.get("wtpsplit_revision") != revision or passed.get("model") != MODEL:
        raise RuntimeError("Smoke test used another bundle, code revision or model")
    if passed.get("runtime", {}).get("wtpsplit") != runtime["wtpsplit"] or \
       passed.get("runtime", {}).get("adapters") != runtime["adapters"]:
        raise RuntimeError("LoRA package versions changed since smoke test")
    runtime["base_model_revision"] = cached_hub_revision("segment-any-text/" + MODEL)
    runtime["tokenizer_revision"] = cached_hub_revision("FacebookAI/xlm-roberta-base")
    if passed["runtime"].get("base_model_revision") != runtime["base_model_revision"] or \
       passed["runtime"].get("tokenizer_revision") != runtime["tokenizer_revision"]:
        raise RuntimeError("Hugging Face model/tokenizer revisions changed since smoke test")
    run = root / f"target_{target}"
    run.mkdir(parents=True, exist_ok=True)
    output_dir = run / "trainer"
    config = base_config(bundle, target, output_dir)
    config["seed"] = manifest["seed"]
    adapter = adapter_folder(output_dir)
    if not adapter_exists(adapter):
        checkpoints = sorted(output_dir.glob("checkpoint-*"),
                             key=lambda p: int(p.name.rsplit("-", 1)[-1]))
        if checkpoints:
            config["resume_from_checkpoint"] = str(checkpoints[-1])
        run_trainer(repo, config, run / "config.json")
        if not adapter_exists(adapter):
            raise RuntimeError(f"Trainer finished without a loadable adapter: {adapter}")
    state_path = output_dir / "trainer_state.json"
    if state_path.is_file():
        export_loss_history(state_path, run / "loss_history.csv")
    write_json(run / "training_done.json", {"target": target, "model": MODEL,
               "seed": manifest["seed"], "adapter": str(adapter),
               "wtpsplit_revision": revision,
               "runtime": runtime,
               "bundle_manifest_sha256": digest(bundle / "manifest.json"),
               "config_sha256": digest(run / "config.json"), "completed_at_utc": stamp()})
    evaluate_fold(bundle, target, run, adapter)


def pieces_to_spans(raw: str, pieces: list[str]) -> list[tuple[int, int]]:
    spans = []
    cursor = 0
    for piece in pieces:
        if not piece:
            continue
        start = raw.find(piece, cursor)
        if start < 0 or any(ch not in "\r\n" for ch in raw[cursor:start]):
            raise ValueError(f"Cannot align SaT piece after offset {cursor}")
        end = start + len(piece)
        if not 0 <= start < end <= len(raw) or raw[start:end] != piece:
            raise ValueError("Incorrect SaT offset")
        if spans and start < spans[-1][1]:
            raise ValueError("Overlapping SaT pieces")
        spans.append((start, end))
        cursor = end
    if not spans or any(ch not in "\r\n" for ch in raw[cursor:]):
        raise ValueError("SaT did not cover the text except input newlines")
    return spans


def spans_from_probabilities(raw: str, probabilities, threshold: float) -> list[tuple[int, int]]:
    import numpy as np
    from wtpsplit.utils import indices_to_sentences
    if len(probabilities) != len(raw):
        raise ValueError("SaT probability vector has a different character length")
    sentences = indices_to_sentences(raw, np.where(probabilities > threshold)[0],
                                     strip_whitespace=False)
    pieces = []
    for i, sentence in enumerate(sentences):
        if i < len(sentences) - 1 and sentence.endswith("\n"):
            sentence = sentence[:-1]
        pieces.extend(sentence.split("\n"))
    return pieces_to_spans(raw, pieces)


def check_native_equivalence(sat, samples: list[str], *, check_default: bool = False) -> None:
    for raw in samples:
        probabilities = list(sat.predict_proba([raw], stride=64))[0]
        for threshold in (0.10, DEFAULT, 0.80):
            native = pieces_to_spans(raw, list(sat.split([raw], threshold=threshold))[0])
            derived = spans_from_probabilities(raw, probabilities, threshold)
            if native != derived:
                raise RuntimeError("Cached thresholding differs from native SaT.split")
            if check_default and threshold == DEFAULT:
                implicit = pieces_to_spans(raw, list(sat.split([raw]))[0])
                if implicit != native:
                    raise RuntimeError("The installed SaT default threshold is not 0.25")


def cache_connection(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""CREATE TABLE IF NOT EXISTS predictions (
        model TEXT NOT NULL, corpus TEXT NOT NULL, document_id TEXT NOT NULL,
        spans_json TEXT NOT NULL, PRIMARY KEY(model, corpus, document_id))""")
    db.commit()
    return db


def predict_all(sat, rows: list[dict], db: sqlite3.Connection, model_key: str) -> None:
    from tqdm import tqdm
    for row in tqdm(rows, desc=f"SaT {model_key}", unit="recit"):
        corpus, ident, raw = row["corpus"], row["document_id"], row["raw_text"]
        if db.execute("SELECT 1 FROM predictions WHERE model=? AND corpus=? AND document_id=?",
                      (model_key, corpus, ident)).fetchone():
            continue
        probabilities = list(sat.predict_proba([raw], stride=64))[0]
        spans = {f"{threshold:.2f}": spans_from_probabilities(raw, probabilities, threshold)
                 for threshold in THRESHOLDS}
        with db:
            db.execute("INSERT INTO predictions VALUES(?,?,?,?)",
                       (model_key, corpus, ident, json.dumps(spans)))


def saved_spans(db: sqlite3.Connection, model_key: str, row: dict,
                threshold: float) -> list[tuple[int, int]]:
    found = db.execute("SELECT spans_json FROM predictions WHERE model=? AND corpus=? AND document_id=?",
                       (model_key, row["corpus"], row["document_id"])).fetchone()
    if not found:
        raise ValueError(f"Missing prediction for {model_key}/{row['corpus']}/{row['document_id']}")
    return [tuple(x) for x in json.loads(found[0])[f"{threshold:.2f}"]]


def counts_for(rows: list[dict], db: sqlite3.Connection, model_key: str,
               threshold: float) -> dict:
    counts = collections.Counter()
    role = collections.Counter()
    for row in rows:
        raw = row["raw_text"]
        ref = {(u["start"], u["end"]) for u in row["units"]}
        pred = set(saved_spans(db, model_key, row, threshold))
        counts["tp"] += len(ref & pred)
        counts["pred"] += len(pred)
        counts["ref"] += len(ref)
        ref_cuts = {p for a, b in ref for p in (a, b) if 0 < p < len(raw)}
        pred_cuts = {p for a, b in pred for p in (a, b) if 0 < p < len(raw)}
        counts["boundary_tp"] += len(ref_cuts & pred_cuts)
        counts["boundary_pred"] += len(pred_cuts)
        counts["boundary_ref"] += len(ref_cuts)
        counts["documents"] += 1
        for unit in row["units"]:
            role[f"{unit['role']}_total"] += 1
            role[f"{unit['role']}_exact"] += (unit["start"], unit["end"]) in pred
    return {"documents": counts["documents"], "reference_units": counts["ref"],
            "predicted_segments": counts["pred"],
            "exact_precision": counts["tp"] / counts["pred"] if counts["pred"] else None,
            "exact_recall": counts["tp"] / counts["ref"] if counts["ref"] else None,
            "exact_f1": 2 * counts["tp"] / (counts["pred"] + counts["ref"])
            if counts["pred"] + counts["ref"] else None,
            "boundary_f1": 2 * counts["boundary_tp"] /
            (counts["boundary_pred"] + counts["boundary_ref"])
            if counts["boundary_pred"] + counts["boundary_ref"] else None,
            "role_exact_recall": {r: role[f"{r}_exact"] / role[f"{r}_total"]
                                  if role[f"{r}_total"] else None for r in sorted(ROLES)}}


def select_threshold(rows: list[dict], db: sqlite3.Connection, model_key: str) -> tuple[float, list[dict]]:
    domains = sorted({row["corpus"] for row in rows})
    curve = []
    for threshold in THRESHOLDS:
        per_domain = {domain: counts_for([r for r in rows if r["corpus"] == domain],
                                      db, model_key, threshold)["exact_f1"]
                      for domain in domains}
        if any(score is None for score in per_domain.values()):
            raise ValueError("Empty source validation domain")
        curve.append({"model": model_key, "threshold": threshold,
                      "source_macro_exact_f1": sum(per_domain.values()) / len(per_domain),
                      **{f"f1_{domain}": score for domain, score in per_domain.items()}})
    best = min(curve, key=lambda row: (-row["source_macro_exact_f1"],
                                       abs(row["threshold"] - DEFAULT), row["threshold"]))
    return best["threshold"], curve


def evaluate_fold(bundle: Path, target: str, run: Path, adapter: Path) -> None:
    folder = bundle / "folds" / f"target_{target}"
    validation = read_jsonl(folder / "val.jsonl")
    test = read_jsonl(folder / "test.jsonl")
    samples = list(dict.fromkeys([validation[0]["raw_text"], test[0]["raw_text"],
                                  next((r["raw_text"] for r in test if "\n" in r["raw_text"]),
                                       test[-1]["raw_text"])]))
    db = cache_connection(run / "predictions.sqlite3")
    try:
        for model_key, adapter_path in (("base", None), ("lora", adapter)):
            sat = load_sat(adapter_path)
            check_native_equivalence(sat, samples, check_default=model_key == "base")
            predict_all(sat, validation + test, db, model_key)
            del sat
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        base_threshold, base_curve = select_threshold(validation, db, "base")
        lora_threshold, lora_curve = select_threshold(validation, db, "lora")
        write_csv(run / "source_validation_thresholds.csv", base_curve + lora_curve,
                  ("model", "threshold", "source_macro_exact_f1") +
                  tuple(f"f1_{c}" for c in CORPORA if c != target))
        results = {"target": target, "reference": "AI_DIRECT_PREANNOTATIONS_NOT_HUMAN_GOLD",
                   "model": MODEL, "source_validation": {
                       "base_threshold": base_threshold, "lora_threshold": lora_threshold,
                       "n_documents": len(validation)},
                   "test": {
                       "M0_original": counts_for(test, db, "base", DEFAULT),
                       "M1_calibrated": counts_for(test, db, "base", base_threshold),
                       "M3_lora_calibrated": counts_for(test, db, "lora", lora_threshold)}}
        results["test"]["gain_M3_minus_M1"] = (
            results["test"]["M3_lora_calibrated"]["exact_f1"] -
            results["test"]["M1_calibrated"]["exact_f1"])
        write_json(run / "metrics.json", results)
        print(json.dumps({"target": target, "M0": results["test"]["M0_original"]["exact_f1"],
                          "M1": results["test"]["M1_calibrated"]["exact_f1"],
                          "M3": results["test"]["M3_lora_calibrated"]["exact_f1"]},
                         indent=2))
    finally:
        db.close()


def evaluate_command(args: argparse.Namespace) -> None:
    bundle = Path(args.bundle).resolve()
    manifest = verify_bundle(bundle)
    target = fold_for_index(manifest, args.task_index)
    run = Path(args.run_root).resolve() / f"target_{target}"
    adapter = adapter_folder(run / "trainer")
    if not adapter_exists(adapter):
        raise FileNotFoundError(f"No LoRA adapter: {adapter}")
    evaluate_fold(bundle, target, run, adapter)


def summarize(args: argparse.Namespace) -> None:
    root = Path(args.run_root).resolve()
    rows = []
    for target in CORPORA:
        path = root / f"target_{target}" / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing fold result: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["target"] != target or result["reference"] != "AI_DIRECT_PREANNOTATIONS_NOT_HUMAN_GOLD":
            raise ValueError(f"Unexpected result: {path}")
        rows.append({"target": target, "n_test": result["test"]["M0_original"]["documents"],
                     "M0": result["test"]["M0_original"]["exact_f1"],
                     "M1": result["test"]["M1_calibrated"]["exact_f1"],
                     "M3": result["test"]["M3_lora_calibrated"]["exact_f1"],
                     "M0_boundary": result["test"]["M0_original"]["boundary_f1"],
                     "M1_boundary": result["test"]["M1_calibrated"]["boundary_f1"],
                     "M3_boundary": result["test"]["M3_lora_calibrated"]["boundary_f1"],
                     "gain_M3_minus_M1": result["test"]["gain_M3_minus_M1"]})
    macro = {key: sum(r[key] for r in rows) / len(rows) for key in
             ("M0", "M1", "M3", "M0_boundary", "M1_boundary", "M3_boundary",
              "gain_M3_minus_M1")}
    rows.append({"target": "macro_6_domains", "n_test": sum(r["n_test"] for r in rows), **macro})
    write_csv(root / "summary.csv", rows,
              ("target", "n_test", "M0", "M1", "M3", "M0_boundary",
               "M1_boundary", "M3_boundary", "gain_M3_minus_M1"))
    write_json(root / "summary.json", {"reference": "AI_DIRECT_PREANNOTATIONS_NOT_HUMAN_GOLD",
              "results": rows, "n_positive_gains": sum(r["gain_M3_minus_M1"] > 0 for r in rows[:-1])})
    print(json.dumps({"summary": str(root / "summary.csv"), "macro": macro}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "prepare"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--annotation-root", default=str(ROOT / "annotation"))
        cmd.add_argument("--allow-fixture-counts", action="store_true", help=argparse.SUPPRESS)
        if name == "preflight":
            cmd.add_argument("--report", help="Write a private preflight report, even if one corpus blocks preparation")
        if name == "prepare":
            cmd.add_argument("--out", required=True)
            cmd.add_argument("--seed", type=int, default=SEED)
        cmd.set_defaults(func=preflight if name == "preflight" else prepare)
    cmd = sub.add_parser("check-bundle")
    cmd.add_argument("--bundle", required=True)
    cmd.set_defaults(func=check_bundle)
    for name, func in (("smoke", smoke), ("fit", fit), ("evaluate", evaluate_command)):
        cmd = sub.add_parser(name)
        cmd.add_argument("--bundle", required=True)
        cmd.add_argument("--run-root", required=True)
        cmd.add_argument("--task-index", type=int, required=True)
        if name != "evaluate":
            cmd.add_argument("--wtpsplit-repo", required=True)
        cmd.set_defaults(func=func)
    cmd = sub.add_parser("summarize")
    cmd.add_argument("--run-root", required=True)
    cmd.set_defaults(func=summarize)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(2, f"ERREUR: {exc}\n")


if __name__ == "__main__":
    main()
