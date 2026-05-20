"""Évalue la géométrie SCGM sur le corpus test (output_test/<corpus>/scgm_text/)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scgm_text.eval_corpus import evaluate_scgm_on_corpus
from safer_core.test_corpus import method_test_results_dir, resolve_test_corpus


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SCGM geometry metrics on test corpus.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--corpus", type=str, default=None, help="Identifiant test_corpora.yaml")
    p.add_argument("--data_csv", type=str, default=None)
    p.add_argument("--emb_csv", type=str, default=None)
    p.add_argument("--label_col", type=str, default="pred_label")
    p.add_argument("--pred_ok_col", type=str, default="pred_ok")
    p.add_argument("--group_col", type=str, default="accident_id")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = resolve_test_corpus(args.corpus)
    data_csv = args.data_csv or str(spec.data_csv)
    emb_csv = args.emb_csv or str(spec.emb_csv)
    metrics_dir = method_test_results_dir("scgm_text", spec.id) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_csv = metrics_dir / "metrics_geometry_test.csv"
    if out_csv.is_file():
        print(f"Déjà présent : {out_csv}")
        return
    data_path = Path(data_csv) if Path(data_csv).is_absolute() else ROOT_DIR / data_csv
    emb_path = Path(emb_csv) if Path(emb_csv).is_absolute() else ROOT_DIR / emb_csv
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
