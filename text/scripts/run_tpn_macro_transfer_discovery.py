#!/usr/bin/env python3
"""Transfert macro TPN (encodeur modulable + adaptateur prototypique) sur corpus test."""

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
from macro_transfer.tpn_encode import (
    resolve_tpn_checkpoint,
    tpn_method_name,
    validate_encoder_name,
)
from macro_transfer.tpn_pipeline import run_tpn_macro_transfer_discovery


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="configs/tpn_macro_transfer.yaml")
    p.add_argument("--corpus", type=str, default=None)
    p.add_argument("--base-method", type=str, default=None, help="softtriple|supcon|batch_triplet|scgm_text")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--source-data-csv", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--emb-csv", type=str, default=None, help="Embeddings Qwen corpus test (SCGM)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--skip-bertopic", action="store_true")
    p.add_argument(
        "--topic-embedding-mode",
        type=str,
        default=None,
        choices=["initial", "adapted", "mixed"],
        help="Espace embeddings BERTopic : initial|adapted|mixed",
    )
    p.add_argument("--topic-alpha", type=float, default=None, help="Alpha pour mode mixed")
    p.add_argument(
        "--run-bertopic-grid",
        action="store_true",
        help="Lance grid search A0/A1 (coûteux)",
    )
    p.add_argument(
        "--grid-macros",
        type=str,
        default="A0,A1",
        help="Macros pour la grid, séparées par virgule",
    )
    p.add_argument(
        "--macro-topic-config",
        type=str,
        default=None,
        help="YAML optionnel remplaçant bertopic.macro_params",
    )
    p.add_argument("--skip-compression-diagnostics", action="store_true")
    p.add_argument(
        "--bertopic-only",
        action="store_true",
        help="Relance BERTopic sur artefacts existants (embeddings + metadata)",
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--encode-batch-size", type=int, default=None)
    return p.parse_args()


def main() -> None:
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
        stream=sys.stdout,
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    args = _parse_args()
    cfg_path = resolve_repo_path(args.config, repo_root=TEXT_ROOT)
    raw = load_yaml(cfg_path)

    corpus_id = args.corpus or raw.get("corpus") or default_test_corpus_id()
    spec, target_data_csv, emb_csv_resolved = resolve_test_paths_from_config(
        {**raw, "corpus": corpus_id}, corpus_id=corpus_id, anchor=TEXT_ROOT
    )
    corpus_id = spec.id

    source_cfg = raw.get("source") or {}
    source_data_csv = args.source_data_csv or source_cfg.get("data_csv")
    if not source_data_csv:
        raise SystemExit("--source-data-csv ou source.data_csv requis dans la config")
    source_data_csv = str(resolve_repo_path(source_data_csv, repo_root=TEXT_ROOT))
    target_data_csv = str(resolve_repo_path(target_data_csv, repo_root=TEXT_ROOT))

    method_cfg = dict(raw.get("method") or {})
    if args.base_method:
        method_cfg["base_method"] = args.base_method
    base_method = validate_encoder_name(method_cfg.get("base_method") or "softtriple")
    method_name = tpn_method_name(base_method)

    checkpoints_block = raw.get("checkpoints") or {}
    base_method_overridden = bool(args.base_method)
    ckpt = resolve_tpn_checkpoint(
        base_method,
        method_cfg,
        checkpoints_block,
        explicit_checkpoint=args.checkpoint,
        base_method_overridden=base_method_overridden,
    )
    ckpt = str(resolve_repo_path(ckpt, repo_root=TEXT_ROOT))
    logging.getLogger(__name__).info(
        "TPN encoder=%s checkpoint=%s (override_cli=%s)",
        base_method,
        ckpt,
        base_method_overridden,
    )

    contrastive_cfg = method_cfg.get("contrastive_config") or raw.get("contrastive_config")
    if contrastive_cfg and base_method in ("softtriple", "supcon", "batch_triplet"):
        cfg_name = Path(str(contrastive_cfg)).stem
        if cfg_name != base_method:
            logging.getLogger(__name__).warning(
                "contrastive_config %s ne correspond pas à base_method=%s",
                contrastive_cfg,
                base_method,
            )

    emb_csv = args.emb_csv or method_cfg.get("emb_csv") or raw.get("emb_csv")
    if emb_csv:
        emb_csv = str(resolve_repo_path(emb_csv, repo_root=TEXT_ROOT))
    elif base_method == "scgm_text":
        emb_csv = str(emb_csv_resolved)

    output_dir = args.output_dir
    if not output_dir:
        output_dir = str(macro_transfer_output_dir(method_name, corpus_id, anchor=TEXT_ROOT))
    else:
        output_dir = str(resolve_repo_path(output_dir, repo_root=TEXT_ROOT))

    raw = {
        **raw,
        "corpus": corpus_id,
        "repo_anchor": str(TEXT_ROOT),
        "method": {**method_cfg, "base_method": base_method},
    }
    if emb_csv:
        raw.setdefault("method", {})["emb_csv"] = emb_csv

    if args.encode_batch_size is not None:
        enc = dict(raw.get("encoding") or {})
        enc["encode_batch_size"] = int(args.encode_batch_size)
        raw["encoding"] = enc
    if args.device:
        enc = dict(raw.get("encoding") or {})
        enc["device"] = args.device
        raw["encoding"] = enc

    grid_macros = None
    if args.grid_macros:
        grid_macros = [m.strip() for m in args.grid_macros.split(",") if m.strip()]

    logging.getLogger(__name__).info(
        "Démarrage TPN : corpus=%s encoder=%s out=%s device=%s epochs=%s",
        corpus_id,
        base_method,
        output_dir,
        args.device or raw.get("encoding", {}).get("device", "cuda"),
        args.epochs or raw.get("tpn", {}).get("epochs", 50),
    )
    manifest = run_tpn_macro_transfer_discovery(
        checkpoint=ckpt,
        source_data_csv=source_data_csv,
        target_data_csv=target_data_csv,
        output_dir=output_dir,
        config=raw,
        skip_bertopic=args.skip_bertopic or bool(raw.get("skip_bertopic", False)),
        bertopic_only=args.bertopic_only,
        topic_embedding_mode=args.topic_embedding_mode,
        topic_alpha=args.topic_alpha,
        run_bertopic_grid=args.run_bertopic_grid,
        grid_macros=grid_macros,
        macro_topic_config_path=args.macro_topic_config,
        skip_compression_diagnostics=args.skip_compression_diagnostics,
        device=args.device or raw.get("encoding", {}).get("device", "cuda"),
        encode_batch_size=int(
            args.encode_batch_size
            or raw.get("encoding", {}).get("encode_batch_size", 8)
        ),
        epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        emb_csv=emb_csv,
    )
    print("OK:", output_dir, f"(corpus={corpus_id}, method={method_name}, encoder={base_method})")
    print("n_target:", manifest.get("n_target"))
    print("accuracy (adapted):", manifest.get("metrics_adapted", {}).get("accuracy"))
    themes_csv = Path(output_dir) / "topics_bertopic" / "themes_by_macro.csv"
    if themes_csv.is_file():
        print("bertopic:", themes_csv)


if __name__ == "__main__":
    main()
