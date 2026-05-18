"""Exporte les embeddings bruts (encodeur figé) + métriques géométrie."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from metrics.geometry import build_geometry_metrics_row
from safer_core.data_loading import load_metadata_with_embeddings
from safer_core.io import ensure_dir, save_metrics_geometry
from safer_core.paths import layout_method_output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export raw encoder embeddings + geometry metrics.")
    p.add_argument("--config", type=str, default="configs/methods/raw_embedding.yaml")
    p.add_argument("--data_csv", type=str, default=None)
    p.add_argument("--emb_csv", type=str, default=None)
    p.add_argument("--label_col", type=str, default="pred_label")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--method_name", type=str, default=None, help="Libellé ligne métriques (method).")
    p.add_argument("--method_slug", type=str, default="raw_embedding", help="Slug layout_method_output.")
    p.add_argument("--skip_npy", action="store_true", help="Ne pas écrire all_embeddings.npy/.csv")
    return p.parse_args()


def _apply_yaml_config(args: argparse.Namespace) -> None:
    cfg_path = ROOT_DIR / args.config
    if not cfg_path.is_file():
        return
    with cfg_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if args.data_csv is None:
        args.data_csv = cfg.get("dataset_path") or cfg.get("data_csv") or "dataset/data_btp.csv"
    if args.emb_csv is None:
        args.emb_csv = cfg.get("emb_csv") or "embeddings/Qwen3-Embedding-0.6B_btp.csv"
    if args.output_dir is None:
        args.output_dir = cfg.get("output_dir") or "resultats/raw_embedding"
    if args.label_col == "pred_label" and cfg.get("label_col"):
        args.label_col = cfg["label_col"]
    if args.method_name is None:
        args.method_name = cfg.get("method_display_name") or cfg.get("method_name") or "Embedding brut"
    if args.method_slug == "raw_embedding" and cfg.get("method_slug"):
        args.method_slug = cfg["method_slug"]


def main() -> None:
    args = parse_args()
    _apply_yaml_config(args)
    if args.data_csv is None:
        args.data_csv = "dataset/data_btp.csv"
    if args.emb_csv is None:
        args.emb_csv = "embeddings/Qwen3-Embedding-0.6B_btp.csv"
    if args.output_dir is None:
        args.output_dir = "resultats/raw_embedding"
    if args.method_name is None:
        args.method_name = "Embedding brut"

    layout = layout_method_output(args.method_slug, args.output_dir)
    for key in ("embeddings", "metrics"):
        ensure_dir(layout[key])

    merged, dim_cols = load_metadata_with_embeddings(
        args.data_csv,
        args.emb_csv,
        label_col=args.label_col,
    )
    emb = merged[dim_cols].to_numpy(dtype=np.float32)

    if not args.skip_npy:
        np.save(layout["embeddings"] / "all_embeddings.npy", emb)
        merged[["doc_id", *dim_cols]].to_csv(
            layout["embeddings"] / "all_embeddings.csv",
            index=False,
        )

    row = build_geometry_metrics_row(
        emb,
        merged[args.label_col].to_numpy(),
        method=str(args.method_name),
    )
    save_metrics_geometry(row, layout["metrics"])
    print(f"Métriques : {layout['metrics'] / 'metrics_geometry.csv'}")
    if not args.skip_npy:
        print(f"Embeddings : {layout['embeddings']} ({emb.shape[0]} lignes)")


if __name__ == "__main__":
    main()
