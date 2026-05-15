"""Run directories and structured training logs."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Iterable, List, Optional

from scgm_text.utils_io import ensure_dir


def create_run_dirs(output_dir: str) -> Dict[str, str]:
    metrics_dir = ensure_dir(os.path.join(output_dir, "metrics"))
    return {
        "output_dir": output_dir,
        "metrics_dir": metrics_dir,
        "train_log_csv": os.path.join(metrics_dir, "train_log.csv"),
        "epoch_jsonl": os.path.join(metrics_dir, "epoch_metrics.jsonl"),
        "legacy_logs_csv": os.path.join(output_dir, "logs.csv"),
    }


def append_jsonl(record: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_metrics_csv(rows: List[Dict[str, Any]], path: str, fieldnames: Optional[Iterable[str]] = None) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    if not rows:
        return
    keys = list(fieldnames) if fieldnames else list(rows[0].keys())
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def init_metrics_csv(path: str, fieldnames: List[str]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
