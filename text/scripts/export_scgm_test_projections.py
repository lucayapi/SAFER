"""Exporte les embeddings SCGM projetés sur le corpus test (output_test/<corpus>/scgm_text/)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from safer_core.test_corpus import method_test_results_dir, resolve_test_corpus
from scgm_text.eval_corpus import save_scgm_projected_corpus


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export projections SCGM sur corpus test.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="output/scgm_text/checkpoints/best_model.pt",
    )
    p.add_argument("--corpus", type=str, default=None)
    p.add_argument("--data_csv", type=str, default=None)
    p.add_argument("--emb_csv", type=str, default=None)
    p.add_argument("--label_col", type=str, default="pred_label")
    p.add_argument("--group_col", type=str, default="accident_id")
    p.add_argument("--pred_ok_col", type=str, default="pred_ok")
    p.add_argument("--text_col", type=str, default="sentence")
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=512)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = resolve_test_corpus(args.corpus)
    data_csv = args.data_csv or str(spec.data_csv)
    emb_csv = args.emb_csv or str(spec.emb_csv)
    emb_dir = method_test_results_dir("scgm_text", spec.id) / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    paths = save_scgm_projected_corpus(
        args.checkpoint,
        data_csv,
        emb_csv,
        emb_dir,
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
