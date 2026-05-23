#!/usr/bin/env python3
"""Vérifie PKBatchSampler sur dataset/data_btp.csv (même filtrage que l'entraînement)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.config import load_contrastive_config
from contrastive_methods.data import prepare_text_dataset, split_train_val, train_val_metadata
from contrastive_methods.samplers.pk_batch_sampler import (
    PKBatchSampler,
    debug_pk_sampler_batches,
    resolve_batch_triplet_pk_params,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug PKBatchSampler sur data_btp.")
    parser.add_argument(
        "--config",
        default="configs/methods/batch_triplet.yaml",
        help="Config batch_triplet YAML.",
    )
    parser.add_argument("--n-batches", type=int, default=10)
    args = parser.parse_args()

    cfg = load_contrastive_config("batch_triplet", config_path=TEXT_ROOT / args.config)
    dataset = prepare_text_dataset(cfg)
    train_idx, val_idx = split_train_val(dataset, cfg)
    train_df, _ = train_val_metadata(dataset, train_idx, val_idx)
    labels = train_df["label_id"].astype(int).tolist()

    pk_params = resolve_batch_triplet_pk_params(cfg, labels)
    sampler = PKBatchSampler(
        labels=labels,
        batch_size=pk_params.batch_size,
        classes_per_batch=pk_params.classes_per_batch,
        samples_per_class=pk_params.samples_per_class,
        seed=pk_params.seed,
    )
    debug_pk_sampler_batches(
        sampler,
        labels,
        n_batches=args.n_batches,
        expected_classes=pk_params.classes_per_batch,
        expected_samples_per_class=pk_params.samples_per_class,
    )
    print(f"OK — {args.n_batches} batchs vérifiés (P={pk_params.classes_per_batch}, K={pk_params.samples_per_class}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
