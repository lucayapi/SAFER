#!/usr/bin/env python3
"""Transfert macro TPN full-encoder (end-to-end, classifieur prototypique)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from macro_transfer.encode import load_target_metadata
from macro_transfer.tpn_encode import resolve_tpn_checkpoint, tpn_method_name, validate_encoder_name
from macro_transfer.tpn_full_encoder import FullEncoderTPNModel, encode_texts_corpus, train_tpn_full_encoder
from macro_transfer.tpn_topics_phase import merge_bertopic_cfg, run_bertopic_phase
from safer_core.io import load_yaml
from safer_core.paths import resolve_repo_path
from safer_core.test_corpus import default_test_corpus_id, resolve_test_paths_from_config
from scgm_text.dataset_text_embeddings import load_filtered_metadata


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="configs/tpn_macro_transfer.yaml")
    p.add_argument("--corpus", type=str, default=None)
    p.add_argument("--base-method", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--backbone-name", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--skip-bertopic", action="store_true")
    p.add_argument("--prototype-mode", type=str, choices=["batch", "ema_global"], default=None)
    p.add_argument("--pseudo-label-threshold", type=float, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def _resolve_output_dir(base_method: str, corpus: str, output_dir: Optional[str]) -> Path:
    if output_dir:
        return resolve_repo_path(output_dir, repo_root=TEXT_ROOT)
    return resolve_repo_path(
        Path("output_test") / corpus / "macro_transfer" / f"tpn_full_{base_method}",
        repo_root=TEXT_ROOT,
    )


def _apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    out = dict(cfg)
    method = dict(out.get("method") or {})
    full = dict(out.get("full_encoder") or {})
    tpn = dict(out.get("tpn") or {})
    encoding = dict(out.get("encoding") or {})

    if args.base_method:
        method["base_method"] = args.base_method
    if args.backbone_name:
        method["backbone_name"] = args.backbone_name
    if args.epochs is not None:
        full["epochs"] = int(args.epochs)
    if args.lr is not None:
        full["learning_rate"] = float(args.lr)
    if args.device:
        encoding["device"] = args.device
    if args.prototype_mode:
        full["prototype_mode"] = args.prototype_mode
    if args.pseudo_label_threshold is not None:
        tpn["pseudo_label_threshold"] = float(args.pseudo_label_threshold)
    out["method"] = method
    out["full_encoder"] = full
    out["tpn"] = tpn
    out["encoding"] = encoding
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    args = _parse_args()
    cfg_path = resolve_repo_path(args.config, repo_root=TEXT_ROOT)
    cfg = load_yaml(cfg_path)
    cfg = _apply_cli_overrides(cfg, args)
    cfg["repo_anchor"] = str(TEXT_ROOT)

    corpus = args.corpus or cfg.get("corpus") or default_test_corpus_id()
    spec, target_data_csv, _emb_csv = resolve_test_paths_from_config(
        {**cfg, "corpus": corpus},
        corpus_id=corpus,
        anchor=TEXT_ROOT,
    )
    corpus = spec.id
    cfg["corpus"] = corpus

    method_cfg = dict(cfg.get("method") or {})
    base_method = validate_encoder_name(method_cfg.get("base_method") or "scgm_text")
    checkpoints_block = cfg.get("checkpoints") or {}
    checkpoint = resolve_tpn_checkpoint(
        base_method,
        method_cfg,
        checkpoints_block,
        explicit_checkpoint=args.checkpoint,
        base_method_overridden=bool(args.base_method),
    )
    checkpoint = str(resolve_repo_path(checkpoint, repo_root=TEXT_ROOT))
    backbone_name = str(method_cfg.get("backbone_name") or "")

    out_dir = _resolve_output_dir(base_method, corpus, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_cfg = dict(cfg.get("source") or {})
    target_cfg = dict(cfg.get("target") or {})
    text_col_s = source_cfg.get("text_col", "sentence")
    text_col_t = target_cfg.get("text_col", "sentence")
    label_col_s = source_cfg.get("label_col", "pred_label")
    pred_ok_col_s = source_cfg.get("pred_ok_col", "pred_ok")
    group_col_s = source_cfg.get("group_col", "accident_id")
    target_label_col = target_cfg.get("label_col", "pred_label")
    target_pred_ok_col = target_cfg.get("pred_ok_col", "pred_ok")

    source_data_csv = source_cfg.get("data_csv")
    if not source_data_csv:
        raise SystemExit("source.data_csv requis")
    source_data_csv = str(resolve_repo_path(source_data_csv, repo_root=TEXT_ROOT))
    target_data_csv = str(resolve_repo_path(target_data_csv, repo_root=TEXT_ROOT))

    source_df = load_filtered_metadata(
        source_data_csv,
        label_col=label_col_s,
        pred_ok_col=pred_ok_col_s,
        group_col=group_col_s,
        text_col=text_col_s,
    )
    target_df = load_target_metadata(target_data_csv, text_col=text_col_t)

    enc_cfg = dict(cfg.get("encoding") or {})
    full_cfg = dict(cfg.get("full_encoder") or {})
    tpn_cfg = dict(cfg.get("tpn") or {})
    loss_weights = dict(cfg.get("loss_weights") or {})
    if "reg" not in loss_weights and "preserve" in loss_weights:
        loss_weights["reg"] = loss_weights["preserve"]
    if "reg" not in loss_weights:
        loss_weights["reg"] = 0.0

    model = FullEncoderTPNModel(
        base_method=base_method,
        checkpoint=checkpoint,
        backbone_name=backbone_name,
        max_seq_length=int(enc_cfg.get("max_seq_length", 256)),
        pooling=str(enc_cfg.get("pooling", "mean")),
        freeze_backbone=bool(full_cfg.get("freeze_backbone", False)),
        device=str(enc_cfg.get("device", args.device or "cuda")),
    )

    # Embeddings init pour analyses/BERTopic mixed initial vs adapted
    init_dir = out_dir / "embeddings"
    init_dir.mkdir(parents=True, exist_ok=True)
    source_init = encode_texts_corpus(
        model,
        source_df[text_col_s].astype(str).tolist(),
        batch_size=int(full_cfg.get("source_batch_size", 16)),
        log_label="source_initial",
    )
    target_init = encode_texts_corpus(
        model,
        target_df[text_col_t].astype(str).tolist(),
        batch_size=int(full_cfg.get("target_batch_size", 16)),
        log_label="target_initial",
    )
    np.save(init_dir / "source_projected.npy", source_init)
    np.save(init_dir / "target_projected.npy", target_init)

    result = train_tpn_full_encoder(
        model=model,
        source_df=source_df,
        target_df=target_df,
        out_dir=out_dir,
        tpn_cfg=tpn_cfg,
        full_cfg={
            **full_cfg,
            "source_text_col": text_col_s,
            "target_text_col": text_col_t,
        },
        loss_weights=loss_weights,
        label_col=label_col_s,
        target_label_col=target_label_col,
        pred_ok_col_target=target_pred_ok_col,
    )

    skip_bertopic = bool(args.skip_bertopic or cfg.get("skip_bertopic", False))
    bertopic_summary = {}
    if not skip_bertopic:
        bertopic_cfg = merge_bertopic_cfg(
            cfg.get("bertopic") or {},
            None,
            TEXT_ROOT,
        )
        topics_export_cfg = cfg.get("topics_export") or {}
        metadata_probs = pd.read_csv(out_dir / "transfer" / "metadata_with_tpn_full_macro_probs.csv")
        prob_cols = [f"p_{m}" for m in ("A0", "A1", "B", "C")]
        gating_cols = ["m_hat", "ambiguous", "q_conf", "margin"] + prob_cols
        gating = metadata_probs[gating_cols].copy()
        target_adapted = np.load(out_dir / "embeddings" / "target_full_embeddings.npy")
        bertopic_summary = run_bertopic_phase(
            out=out_dir,
            meta_t=target_df,
            gating_adapted=gating,
            h_t=target_init,
            h_t_adapted=target_adapted,
            method_name=f"tpn_full_{base_method}",
            bertopic_cfg=bertopic_cfg,
            topics_export_cfg=topics_export_cfg,
            text_col_t=text_col_t,
            repo_anchor=TEXT_ROOT,
            corpus_id=corpus,
            topic_embedding_mode=None,
            topic_alpha=None,
            run_bertopic_grid=False,
            grid_macros=None,
            skip_compression_diagnostics=False,
        )

    manifest = {
        "method": f"tpn_full_{base_method}",
        "base_method": base_method,
        "checkpoint": checkpoint,
        "backbone_name": model.backbone_name,
        "output_dir": str(out_dir),
        "source_data_csv": source_data_csv,
        "target_data_csv": target_data_csv,
        "skip_bertopic": skip_bertopic,
        "bertopic_summary": bertopic_summary,
        "train_result": result,
    }
    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("OK:", out_dir)
    print("method:", manifest["method"])
    print("metrics:", out_dir / "transfer" / "metrics_tpn_full.json")


if __name__ == "__main__":
    main()
