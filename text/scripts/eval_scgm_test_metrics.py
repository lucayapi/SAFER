"""Évalue la géométrie SCGM sur le corpus test (metrics_geometry_test.csv)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scgm_text.eval_corpus import evaluate_scgm_on_corpus
from safer_core.paths import layout_method_output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SCGM geometry metrics on test corpus.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="resultats/scgm_text")
    p.add_argument("--data_csv", type=str, default="dataset/test/data_metallurgie.csv")
    p.add_argument("--emb_csv", type=str, default="embeddings/test/Qwen3-Embedding-0.6B_metallurgie.csv")
    p.add_argument("--label_col", type=str, default="pred_label")
    p.add_argument("--pred_ok_col", type=str, default="pred_ok")
    p.add_argument("--group_col", type=str, default="accident_id")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    layout = layout_method_output("scgm_text", args.output_dir)
    metrics_dir = layout["metrics"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_csv = metrics_dir / "metrics_geometry_test.csv"
    if out_csv.is_file():
        print(f"Déjà présent : {out_csv}")
        return
    data_path = ROOT_DIR / args.data_csv
    emb_path = ROOT_DIR / args.emb_csv
    if not data_path.is_file():
        raise FileNotFoundError(f"data_csv absent : {data_path}")
    if not emb_path.is_file():
        raise FileNotFoundError(f"emb_csv absent : {emb_path}")
    evaluate_scgm_on_corpus(
        args.checkpoint,
        str(data_path),
        str(emb_path),
        corpus="test",
        metrics_dir=metrics_dir,
        label_col=args.label_col,
        pred_ok_col=args.pred_ok_col,
        group_col=args.group_col,
    )
    print(f"Écrit : {out_csv}")


if __name__ == "__main__":
    main()
