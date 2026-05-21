#!/usr/bin/env python3
"""Transfert macro TPN (SoftTriple + adaptateur prototypique) sur corpus test."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from safer_core.io import load_yaml
from safer_core.paths import resolve_repo_path
from safer_core.test_corpus import (
    default_test_corpus_id,
    macro_transfer_output_dir,
    resolve_test_paths_from_config,
)
from macro_transfer.tpn_pipeline import METHOD_NAME, run_tpn_macro_transfer_discovery


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="configs/tpn_macro_transfer.yaml")
    p.add_argument("--corpus", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--source-data-csv", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--skip-bertopic", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--encode-batch-size", type=int, default=None)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    cfg_path = resolve_repo_path(args.config, repo_root=TEXT_ROOT)
    raw = load_yaml(cfg_path)

    corpus_id = args.corpus or raw.get("corpus") or default_test_corpus_id()
    spec, target_data_csv, _emb_csv = resolve_test_paths_from_config(
        {**raw, "corpus": corpus_id}, corpus_id=corpus_id, anchor=TEXT_ROOT
    )
    corpus_id = spec.id

    source_cfg = raw.get("source") or {}
    source_data_csv = args.source_data_csv or source_cfg.get("data_csv")
    if not source_data_csv:
        raise SystemExit("--source-data-csv ou source.data_csv requis dans la config")
    source_data_csv = str(resolve_repo_path(source_data_csv, repo_root=TEXT_ROOT))
    target_data_csv = str(resolve_repo_path(target_data_csv, repo_root=TEXT_ROOT))

    method_cfg = raw.get("method") or {}
    ckpt = args.checkpoint or method_cfg.get("checkpoint")
    if not ckpt:
        raise SystemExit("--checkpoint ou method.checkpoint requis")
    ckpt = str(resolve_repo_path(ckpt, repo_root=TEXT_ROOT))

    output_dir = args.output_dir
    if not output_dir:
        output_dir = str(macro_transfer_output_dir(METHOD_NAME, corpus_id, anchor=TEXT_ROOT))
    else:
        output_dir = str(resolve_repo_path(output_dir, repo_root=TEXT_ROOT))

    raw = {**raw, "corpus": corpus_id, "repo_anchor": str(TEXT_ROOT)}
    if args.encode_batch_size is not None:
        enc = dict(raw.get("encoding") or {})
        enc["encode_batch_size"] = int(args.encode_batch_size)
        raw["encoding"] = enc
    if args.device:
        enc = dict(raw.get("encoding") or {})
        enc["device"] = args.device
        raw["encoding"] = enc

    manifest = run_tpn_macro_transfer_discovery(
        checkpoint=ckpt,
        source_data_csv=source_data_csv,
        target_data_csv=target_data_csv,
        output_dir=output_dir,
        config=raw,
        skip_bertopic=args.skip_bertopic or bool(raw.get("skip_bertopic", False)),
        device=args.device or raw.get("encoding", {}).get("device", "cuda"),
        encode_batch_size=int(
            args.encode_batch_size
            or raw.get("encoding", {}).get("encode_batch_size", 8)
        ),
        epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
    )
    print("OK:", output_dir, f"(corpus={corpus_id})")
    print("n_target:", manifest.get("n_target"))
    print("accuracy (adapted):", manifest.get("metrics_adapted", {}).get("accuracy"))
    themes_csv = Path(output_dir) / "topics_bertopic" / "themes_by_macro.csv"
    if themes_csv.is_file():
        print("bertopic:", themes_csv)


if __name__ == "__main__":
    main()
