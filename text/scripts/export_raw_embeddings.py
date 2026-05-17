"""Exporte les embeddings bruts (encodeur figé) + métriques géométrie."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from metrics.geometry import build_geometry_metrics_row
from safer_core.data_loading import load_metadata_with_embeddings
from safer_core.io import save_metrics_geometry
from safer_core.paths import layout_method_output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export raw encoder embeddings.")
    p.add_argument("--config", type=str, default="configs/methods/raw_embedding.yaml")
    p.add_argument("--data_csv", type=str, default="dataset/data_btp.csv")
    p.add_argument("--emb_csv", type=str, default="embeddings/Qwen3-Embedding-0.6B_btp.csv")
    p.add_argument("--label_col", type=str, default="pred_label")
    p.add_argument("--output_dir", type=str, default="resultats/raw_embedding")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    layout = layout_method_output("raw_embedding", args.output_dir)
    merged, dim_cols = load_metadata_with_embeddings(
        args.data_csv,
        args.emb_csv,
        label_col=args.label_col,
    )
    emb = merged[dim_cols].to_numpy(dtype=np.float32)
    np.save(layout["embeddings"] / "all_embeddings.npy", emb)
    merged[["doc_id", *dim_cols]].to_csv(
        layout["embeddings"] / "all_embeddings.csv",
        index=False,
    )

    row = build_geometry_metrics_row(
        emb,
        merged[args.label_col].to_numpy(),
        method="Embedding brut",
    )
    save_metrics_geometry(row, layout["metrics"])
    print(f"Exporté : {layout['embeddings']} ({emb.shape[0]} lignes)")


if __name__ == "__main__":
    main()
