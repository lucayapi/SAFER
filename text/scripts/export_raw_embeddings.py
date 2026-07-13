"""Exporte les embeddings bruts (encodeur figé) + évaluation classification LR."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from safer_core.classification_eval import (
    build_cv_summary_from_kfold,
    export_projected_embeddings,
    fit_logistic_on_embeddings,
    evaluate_classifier_on_embeddings,
    resolve_test_corpora,
    save_classification_outputs,
)
from safer_core.data_loading import load_metadata_with_embeddings
from safer_core.io import ensure_dir
from safer_core.paths import layout_method_output
from safer_core.test_corpus import raw_embedding_test_dir, resolve_test_corpus
from scgm_text.dataset_text_embeddings import LABEL2ID


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export raw encoder embeddings + classification eval.")
    p.add_argument("--config", type=str, default="configs/methods/raw_embedding.yaml")
    p.add_argument("--corpus", type=str, default=None, help="test_corpora.yaml (configs raw_embedding_test)")
    p.add_argument("--data_csv", type=str, default=None)
    p.add_argument("--emb_csv", type=str, default=None)
    p.add_argument("--label_col", type=str, default="pred_label")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--method_name", type=str, default=None, help="Libellé ligne métriques (method).")
    p.add_argument("--method_slug", type=str, default="raw_embedding", help="Slug layout_method_output.")
    p.add_argument("--skip_npy", action="store_true", help="Ne pas écrire all_embeddings.npy/.csv")
    p.add_argument("--eval-only", action="store_true", help="Classification seulement (embeddings déjà exportés).")
    return p.parse_args()


def _apply_yaml_config(args: argparse.Namespace) -> None:
    cfg_path = ROOT_DIR / args.config
    if not cfg_path.is_file():
        return
    with cfg_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    if cfg.get("test_corpora") or os.environ.get("TEST_CORPORA"):
        pass
    elif cfg.get("test_corpus") or args.corpus:
        spec = resolve_test_corpus(args.corpus or cfg.get("test_corpus"))
        if args.data_csv is None:
            args.data_csv = str(spec.data_csv)
        if args.emb_csv is None:
            args.emb_csv = str(spec.emb_csv)
        if args.output_dir is None:
            args.output_dir = str(raw_embedding_test_dir(spec.id))
    else:
        if args.data_csv is None:
            args.data_csv = cfg.get("dataset_path") or cfg.get("data_csv") or "dataset/data_btp.csv"
        if args.emb_csv is None:
            args.emb_csv = cfg.get("emb_csv") or "embeddings/Qwen3-Embedding-0.6B_btp.csv"
    if args.output_dir is None:
        args.output_dir = cfg.get("output_dir") or "output/raw_embedding"
    if args.label_col == "pred_label" and cfg.get("label_col"):
        args.label_col = cfg["label_col"]
    if args.method_name is None:
        args.method_name = cfg.get("method_display_name") or cfg.get("method_name") or "Embedding brut"
    if args.method_slug == "raw_embedding" and cfg.get("method_slug"):
        args.method_slug = cfg["method_slug"]


def _export_corpus(
    data_csv: str,
    emb_csv: str,
    emb_dir: Path,
    stem: str,
    label_col: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    merged, dim_cols = load_metadata_with_embeddings(data_csv, emb_csv, label_col=label_col)
    emb = merged[dim_cols].to_numpy(dtype=np.float64)
    export_projected_embeddings(
        emb,
        merged,
        emb_dir,
        stem,
        label_col=label_col,
        group_col="accident_id" if "accident_id" in merged.columns else None,
        text_col="sentence" if "sentence" in merged.columns else None,
    )
    return emb, merged


def main() -> None:
    args = parse_args()
    _apply_yaml_config(args)
    if args.data_csv is None:
        args.data_csv = "dataset/data_btp.csv"
    if args.emb_csv is None:
        args.emb_csv = "embeddings/Qwen3-Embedding-0.6B_btp.csv"
    if args.output_dir is None:
        args.output_dir = "output/raw_embedding"
    if args.method_name is None:
        args.method_name = "Embedding brut"

    layout = layout_method_output(args.method_slug, args.output_dir)
    for key in ("embeddings", "metrics"):
        ensure_dir(layout[key])
    emb_dir = Path(layout["embeddings"])

    cfg_path = ROOT_DIR / args.config
    cfg = {}
    if cfg_path.is_file():
        with cfg_path.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
    test_corpora = resolve_test_corpora(cfg)

    X_btp, btp_meta = _export_corpus(args.data_csv, args.emb_csv, emb_dir, "btp", args.label_col)
    if not args.skip_npy:
        np.save(emb_dir / "all_embeddings.npy", X_btp.astype(np.float32))
        dim_cols = [c for c in btp_meta.columns if c.startswith("dim_")]
        if dim_cols:
            btp_meta[["doc_id", *dim_cols]].to_csv(emb_dir / "all_embeddings.csv", index=False)

    y_train = btp_meta[args.label_col].astype(str).map(LABEL2ID).astype(int).to_numpy()
    pipe = fit_logistic_on_embeddings(X_btp, y_train)
    metrics_by_corpus = {
        "btp": evaluate_classifier_on_embeddings(
            pipe, X_btp, btp_meta[args.label_col].astype(str).to_numpy()
        ),
    }

    for corpus_id in test_corpora:
        try:
            spec = resolve_test_corpus(corpus_id)
            X_test, test_meta = _export_corpus(
                str(spec.data_csv),
                str(spec.emb_csv),
                emb_dir,
                str(corpus_id),
                args.label_col,
            )
            metrics_by_corpus[str(corpus_id)] = evaluate_classifier_on_embeddings(
                pipe,
                X_test,
                test_meta[args.label_col].astype(str).to_numpy(),
            )
        except Exception as exc:
            print(f"[raw_embedding] corpus {corpus_id} ignoré : {exc}", flush=True)

    cv_summary = pd.DataFrame()
    kfold_path = Path(layout["metrics"]) / "kfold_summary.csv"
    if kfold_path.is_file():
        cv_summary = build_cv_summary_from_kfold(pd.read_csv(kfold_path), model_name=args.method_slug)

    paths = save_classification_outputs(
        Path(layout["root"]),
        method_name=str(args.method_slug),
        metrics_by_corpus=metrics_by_corpus,
        cv_summary=cv_summary,
    )
    print(f"Métriques classification : {paths.get('btp')}")
    if paths.get("cross_domain"):
        print(f"Cross-domain : {paths['cross_domain']}")


if __name__ == "__main__":
    main()
