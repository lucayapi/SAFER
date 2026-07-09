"""Exporte les embeddings Qwen figés pour un corpus de test (registre test_corpora.yaml)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.data import prepare_text_dataset
from contrastive_methods.export import export_text_embeddings
from safer_core.test_corpus import default_test_corpus_id, resolve_test_corpus


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export embeddings Qwen pour dataset/test/")
    p.add_argument(
        "--corpus",
        type=str,
        default=None,
        help=f"Identifiant registre test_corpora.yaml (défaut : {default_test_corpus_id()})",
    )
    p.add_argument("--data_csv", type=str, default=None)
    p.add_argument("--output_csv", type=str, default=None)
    p.add_argument("--backbone_name", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_seq_length", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = resolve_test_corpus(args.corpus)
    data_csv = args.data_csv or str(spec.data_csv)
    output_csv = args.output_csv or str(spec.emb_csv)
    cfg = ContrastiveConfig(
        method_name="raw_export",
        dataset_path=Path(data_csv) if Path(data_csv).is_absolute() else ROOT_DIR / data_csv,
        backbone_name=args.backbone_name,
        encode_batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        use_projector=False,
        backbone_trainable=False,
    )
    dataset = prepare_text_dataset(cfg)
    dest = Path(output_csv) if Path(output_csv).is_absolute() else ROOT_DIR / output_csv
    dest.parent.mkdir(parents=True, exist_ok=True)
    export_text_embeddings(
        cfg,
        dataset,
        dest,
        batch_size=cfg.encode_batch_size,
        show_progress=True,
    )
    print(f"Exporté : {dest} ({len(dataset)} lignes, corpus={spec.id})")


if __name__ == "__main__":
    main()
