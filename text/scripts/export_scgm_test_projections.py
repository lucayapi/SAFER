"""Exporte les embeddings SCGM projetés sur le corpus test (sans réentraînement)."""

from __future__ import annotations

import argparse
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from safer_core.paths import layout_method_output
from scgm_text.eval_corpus import save_scgm_projected_corpus


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export projections SCGM sur corpus test.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="resultats/scgm_text/checkpoints/best_model.pt",
    )
    p.add_argument("--output_dir", type=str, default="resultats/scgm_text")
    p.add_argument("--data_csv", type=str, default="dataset/test/data_metallurgie.csv")
    p.add_argument(
        "--emb_csv",
        type=str,
        default="embeddings/test/Qwen3-Embedding-0.6B_metallurgie.csv",
    )
    p.add_argument("--label_col", type=str, default="pred_label")
    p.add_argument("--group_col", type=str, default="accident_id")
    p.add_argument("--pred_ok_col", type=str, default="pred_ok")
    p.add_argument("--text_col", type=str, default="sentence")
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=512)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    layout = layout_method_output("scgm_text", args.output_dir)
    paths = save_scgm_projected_corpus(
        args.checkpoint,
        args.data_csv,
        args.emb_csv,
        layout["embeddings"],
        stem="test",
        label_col=args.label_col,
        pred_ok_col=args.pred_ok_col,
        group_col=args.group_col,
        text_col=args.text_col,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )
    print(f"Exporté : {paths['projections']}")
    print(f"Métadonnées : {paths['metadata']}")


if __name__ == "__main__":
    main()
