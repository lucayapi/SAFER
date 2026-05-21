#!/usr/bin/env python3
"""Transfert macro-guidé + découverte topics intra-macro (corpus test configurable)."""

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
from macro_transfer.pipeline import run_macro_transfer_discovery


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=("scgm_text", "softtriple"), required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Identifiant dans configs/test_corpora.yaml (défaut : registre default)",
    )
    p.add_argument("--config", type=str, default="configs/macro_transfer.yaml")
    p.add_argument("--data-csv", type=str, default=None)
    p.add_argument("--emb-csv", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--confidence-threshold", type=float, default=None)
    p.add_argument("--macro-temperature", type=float, default=None)
    p.add_argument("--softtriple-gamma", type=float, default=None)
    p.add_argument("--scgm-tau", type=float, default=None)
    p.add_argument("--skip-bertopic", action="store_true", help="Désactive BERTopic intra-macro (debug)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None)
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
    if args.data_csv:
        raw = {**raw, "data_csv": args.data_csv}
    if args.emb_csv:
        raw = {**raw, "emb_csv": args.emb_csv}
    spec, data_csv, emb_csv = resolve_test_paths_from_config(raw, corpus_id=corpus_id, anchor=TEXT_ROOT)
    corpus_id = spec.id

    method = args.method
    ckpt = args.checkpoint or raw.get("checkpoints", {}).get(method)
    if not ckpt:
        raise SystemExit(f"--checkpoint ou checkpoints.{method} requis")
    ckpt = str(resolve_repo_path(ckpt, repo_root=TEXT_ROOT))

    output_dir = args.output_dir
    if not output_dir:
        out_roots = raw.get("output_roots") or {}
        output_dir = out_roots.get(method)
    if not output_dir:
        output_dir = str(macro_transfer_output_dir(method, corpus_id, anchor=TEXT_ROOT))
    else:
        output_dir = str(resolve_repo_path(output_dir, repo_root=TEXT_ROOT))

    raw = {**raw, "corpus": corpus_id, "repo_anchor": str(TEXT_ROOT)}

    manifest = run_macro_transfer_discovery(
        method=method,
        checkpoint=ckpt,
        data_csv=data_csv,
        emb_csv=emb_csv,
        output_dir=output_dir,
        config=raw,
        confidence_threshold=float(
            args.confidence_threshold
            if args.confidence_threshold is not None
            else raw.get("confidence_threshold", 0.5)
        ),
        macro_temperature=float(
            args.macro_temperature
            if args.macro_temperature is not None
            else raw.get("macro_temperature", 1.0)
        ),
        softtriple_gamma=float(
            args.softtriple_gamma
            if args.softtriple_gamma is not None
            else raw.get("softtriple_gamma", 0.1)
        ),
        scgm_tau=args.scgm_tau if args.scgm_tau is not None else raw.get("scgm_tau"),
        skip_bertopic=args.skip_bertopic or bool(raw.get("skip_bertopic", False)),
        device=args.device or raw.get("device", "cuda"),
        batch_size=int(args.batch_size or raw.get("batch_size", 512)),
    )
    print("OK:", output_dir, f"(corpus={corpus_id})")
    print("n_units:", manifest.get("n_units"))
    print("accuracy:", manifest.get("transfer_metrics", {}).get("accuracy"))
    themes_csv = Path(output_dir) / "topics_bertopic" / "themes_by_macro.csv"
    if themes_csv.is_file():
        print("bertopic:", themes_csv)
    elif not (args.skip_bertopic or bool(raw.get("skip_bertopic", False))):
        print("bertopic: themes_by_macro.csv absent (échec ou skip_bertopic)")


if __name__ == "__main__":
    main()
