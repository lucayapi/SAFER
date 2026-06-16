"""Tableaux récapitulatifs macro_transfer (exports CSV / notebooks FSP)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from macro_transfer.topics_export import build_macro_topic_test_table


def load_macro_topic_stats(out_dir: Path) -> pd.DataFrame:
    """
    Lit summary/macro_topic_stats.csv ou reconstruit depuis manifest + assignments.
    """
    root = Path(out_dir)
    stats_path = root / "summary" / "macro_topic_stats.csv"
    if stats_path.is_file():
        return pd.read_csv(stats_path)

    manifest_path = root / "run_manifest.json"
    macro_counts: Dict[str, Any] = {}
    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        bert = manifest.get("bertopic_summary") or {}
        macro_counts = bert.get("macro_topic_counts") or {}

    topics_dir = root / "topics_bertopic"
    assign_path = topics_dir / "assignments.csv"
    themes_path = topics_dir / "themes_by_macro.csv"
    assignments = pd.read_csv(assign_path) if assign_path.is_file() else pd.DataFrame()
    themes = pd.read_csv(themes_path) if themes_path.is_file() else pd.DataFrame()

    if not macro_counts and assignments.empty:
        return pd.DataFrame(
            columns=[
                "macro",
                "n_units",
                "n_topics",
                "bruit_pct",
                "plus_gros_topic",
                "plus_gros_topic_pct",
            ]
        )
    return build_macro_topic_test_table(macro_counts, assignments, themes)

