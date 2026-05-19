"""Agrège metrics_geometry.csv de chaque méthode."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from metrics.geometry import METRICS_TABLE_COLUMNS
from safer_core.paths import RESULTS_ROOT, ensure_comparisons_dirs


METHOD_DISPLAY = {
    "raw_embedding": "Embedding brut",
    "scgm_text": "SCGM",
    "batch_triplet": "Batch Triplet",
    "softtriple": "SoftTriple",
    "supcon": "SupCon",
    "malt": "MALT",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", type=str, default="resultats")
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def _load_method_row(method_dir: Path) -> dict | None:
    metrics_dir = method_dir / "metrics"
    for name in (
        "metrics_geometry_btp.csv",
        "metrics_geometry.csv",
        "metrics_geometry.json",
    ):
        path = metrics_dir / name
        if path.suffix == ".csv" and path.is_file():
            df = pd.read_csv(path)
            if len(df):
                return df.iloc[0].to_dict()
        if path.suffix == ".json" and path.is_file():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return None


def main() -> None:
    args = parse_args()
    root = (ROOT_DIR / args.results_root).resolve()
    comp = ensure_comparisons_dirs()
    out_path = Path(args.output) if args.output else comp / "tables" / "embedding_geometry_comparison.csv"

    rows = []
    if root.is_dir():
        for method_dir in sorted(root.iterdir()):
            if not method_dir.is_dir() or method_dir.name == "comparisons":
                continue
            row = _load_method_row(method_dir)
            if row is None:
                continue
            key = method_dir.name
            if row.get("method") in (None, "", key):
                row["method"] = METHOD_DISPLAY.get(key, key)
            rows.append(row)

    if not rows:
        print("Aucune métrique trouvée sous", root)
        return

    df = pd.DataFrame(rows)
    if "eta2_macro_balanced_perc" not in df.columns and "delta_macro_pct" in df.columns:
        df["eta2_macro_balanced_perc"] = df["delta_macro_pct"]
    for col in METRICS_TABLE_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan") if col not in ("method", "macros_ignored", "macros_valid") else ""
    df = df[METRICS_TABLE_COLUMNS]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Écrit : {out_path} ({len(df)} méthodes)")


if __name__ == "__main__":
    main()
