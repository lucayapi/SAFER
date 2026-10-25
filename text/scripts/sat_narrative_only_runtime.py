"""Self-contained helpers for the frozen narrative-only LODO folds and SaT run."""

from __future__ import annotations

import csv
import json
import sqlite3
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import sat_lodo as legacy

ROOT = legacy.ROOT
CORPORA = legacy.CORPORA
MODEL = legacy.MODEL
SEED = 20261017
DEFAULT = 0.25
DATASET_NAME = legacy.DATASET_NAME
ROLES = {"A0", "A1", "B", "C", "AUTRE"}
THRESHOLDS = (0.0, 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005,
              0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1,
              0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8, 0.85, 0.9, 0.95,
              0.99, 0.995, 0.999, 1.0)


def digest(path: Path) -> str:
    return legacy.digest(path)


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict]:
    return legacy.read_jsonl(path)


def write_json(path: Path, value: object) -> None:
    legacy.write_json(path, value)


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    legacy.write_csv(path, rows, fields)


def validate_gold(row: dict) -> None:
    raw = row["raw_text"]
    if not isinstance(raw, str) or row["corpus"] not in CORPORA or not isinstance(row["document_id"], str):
        raise ValueError("Invalid cleaned narrative identity or text")
    units = row.get("units")
    if not isinstance(units, list):
        raise ValueError("Cleaned narrative has no unit list")
    spans = []
    cursor = 0
    for unit in units:
        a, b, role = unit.get("start"), unit.get("end"), unit.get("role")
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(raw):
            raise ValueError("Invalid factual-unit offsets")
        if role not in ROLES or a < cursor:
            raise ValueError("Invalid factual-unit role or order")
        cursor = b
        spans.append((a, b))
    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError("Overlapping factual units")
    covered = bytearray(len(raw))
    for a, b in spans:
        covered[a:b] = b"\x01" * (b - a)
    if any(char.isalnum() and not covered[i] for i, char in enumerate(raw)):
        raise ValueError("Informative text is not covered by the factual GOLD units")


def _probability_connection(path: Path, *, create: bool = False) -> sqlite3.Connection:
    db = sqlite3.connect(path, timeout=60)
    if create:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE IF NOT EXISTS probabilities (corpus TEXT, document_id TEXT, "
                   "n_chars INTEGER, compressed BLOB, PRIMARY KEY(corpus,document_id))")
        db.commit()
    return db


def get_probs(db: sqlite3.Connection, row: dict) -> np.ndarray:
    found = db.execute("SELECT n_chars,compressed FROM probabilities WHERE corpus=? AND document_id=?",
                       (row["corpus"], row["document_id"])).fetchone()
    if found is None:
        raise ValueError(f"Missing model probabilities: {row['corpus']}/{row['document_id']}")
    probs = np.frombuffer(zlib.decompress(found[1]), dtype=np.float32)
    if found[0] != len(probs) or len(probs) != len(row["raw_text"]) or not np.isfinite(probs).all():
        raise ValueError("Model probability vector has invalid length or values")
    return probs


def infer_adapted(sat, db: sqlite3.Connection, rows: list[dict], *, chunk_size: int = 16) -> None:
    from tqdm import tqdm

    present = {(corpus, ident) for corpus, ident in db.execute(
        "SELECT corpus,document_id FROM probabilities")}
    todo = [row for row in rows if (row["corpus"], row["document_id"]) not in present]
    with tqdm(total=len(todo), desc="SaT probabilities", unit="narrative") as bar:
        for offset in range(0, len(todo), chunk_size):
            batch = todo[offset:offset + chunk_size]
            calculated = sat.predict_proba([row["raw_text"] for row in batch], stride=64)
            records = []
            for row, probs in zip(batch, calculated, strict=True):
                data = np.asarray(probs, dtype=np.float32)
                if len(data) != len(row["raw_text"]) or not np.isfinite(data).all():
                    raise ValueError(f"Invalid adapted probabilities: {row['corpus']}/{row['document_id']}")
                records.append((row["corpus"], row["document_id"], len(data), zlib.compress(data.tobytes())))
            with db:
                db.executemany("INSERT INTO probabilities VALUES (?,?,?,?)", records)
            bar.update(len(batch))


def verify_native(sat, db: sqlite3.Connection, rows: list[dict], threshold: float) -> None:
    samples = [rows[0], next((row for row in rows if "\n" in row["raw_text"]), rows[-1])]
    for row in samples:
        raw = row["raw_text"]
        native = legacy.pieces_to_spans(raw, sat.split(raw, threshold=threshold))
        cached = legacy.spans_from_probabilities(raw, get_probs(db, row), threshold)
        if native != cached:
            raise AssertionError("Cached SaT probabilities disagree with SaT.split")


def load_pinned_sat(model_dir: Path, tokenizer_dir: Path, adapter: Path | None = None):
    import torch
    from wtpsplit import SaT

    sat = SaT(str(model_dir), tokenizer_name_or_path=str(tokenizer_dir),
              lora_path=str(adapter) if adapter else None)
    if torch.cuda.is_available():
        sat.to("cuda")
    return sat
