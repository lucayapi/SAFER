"""Exporte les embeddings Qwen figés pour le corpus test métallurgie."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from contrastive_methods.data import prepare_text_dataset
from contrastive_methods.export import export_st_embeddings
from contrastive_methods.config import ContrastiveConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export embeddings Qwen pour dataset/test/")
    p.add_argument("--data_csv", type=str, default="dataset/test/data_metallurgie.csv")
    p.add_argument(
        "--output_csv",
        type=str,
        default="embeddings/Qwen3-Embedding-0.6B_metallurgie_test.csv",
    )
    p.add_argument("--backbone_name", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_seq_length", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ContrastiveConfig(
        method_name="raw_export",
        dataset_path=ROOT_DIR / args.data_csv,
        backbone_name=args.backbone_name,
        encode_batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )
    dataset = prepare_text_dataset(cfg)
    dest = ROOT_DIR / args.output_csv
    dest.parent.mkdir(parents=True, exist_ok=True)
    export_st_embeddings(
        cfg.backbone_name,
        dataset,
        dest,
        batch_size=cfg.encode_batch_size,
        show_progress=True,
    )
    print(f"Exporté : {dest} ({len(dataset)} lignes)")


if __name__ == "__main__":
    main()
