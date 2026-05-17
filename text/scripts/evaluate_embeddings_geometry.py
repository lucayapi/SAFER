"""Évalue eta² + RankMe + C1/C10 sur des embeddings et labels."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from metrics.geometry import METRICS_TABLE_COLUMNS, build_geometry_metrics_row
from safer_core.io import save_metrics_geometry
from safer_core.paths import layout_method_output, resolve_output_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Geometry metrics (eta2, RankMe, C1, C10).")
    p.add_argument("--embeddings_npy", type=str, default=None)
    p.add_argument("--embeddings_csv", type=str, default=None)
    p.add_argument("--metadata_csv", type=str, required=True)
    p.add_argument("--label_col", type=str, default="pred_label")
    p.add_argument("--method", type=str, default="method")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--method_name", type=str, default="scgm_text")
    p.add_argument("--l2_normalize", action="store_true")
    return p.parse_args()


def _load_embeddings(args: argparse.Namespace) -> np.ndarray:
    if args.embeddings_npy:
        return np.load(args.embeddings_npy)
    if args.embeddings_csv:
        df = pd.read_csv(args.embeddings_csv)
        dim_cols = sorted(
            [c for c in df.columns if c.startswith("dim_")],
            key=lambda x: int(x.split("_", 1)[1]),
        )
        if not dim_cols:
            raise ValueError("embeddings_csv sans colonnes dim_*")
        return df[dim_cols].to_numpy(dtype=np.float64)
    raise ValueError("Fournir --embeddings_npy ou --embeddings_csv")


def main() -> None:
    args = parse_args()
    meta = pd.read_csv(args.metadata_csv)
    emb = _load_embeddings(args)
    if emb.shape[0] != len(meta):
        raise ValueError(f"Taille embeddings {emb.shape[0]} != metadata {len(meta)}")

    row = build_geometry_metrics_row(
        emb,
        meta[args.label_col].to_numpy(),
        method=args.method,
        l2_normalize=args.l2_normalize,
    )

    out_root = args.output_dir
    if out_root is None:
        layout = layout_method_output(args.method_name)
        out_root = str(layout["metrics"])
    else:
        out_root = str(resolve_output_dir(args.method_name, out_root) / "metrics")

    os.makedirs(out_root, exist_ok=True)
    save_metrics_geometry(row, __import__("pathlib").Path(out_root))
    table = pd.DataFrame([row], columns=METRICS_TABLE_COLUMNS)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
