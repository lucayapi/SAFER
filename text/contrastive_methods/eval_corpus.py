"""Évaluation classification multi-corpus (BTP + OOD) pour modèles contrastifs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.data import prepare_text_dataset
from contrastive_methods.hf_training_common import encode_contrastive_texts, get_device
from contrastive_methods.post_eval import (
    evaluate_classifier_on_embeddings,
    fit_classifier_on_embeddings,
)
from safer_core.classification_eval import (
    build_cv_summary_from_kfold,
    export_projected_embeddings,
    resolve_test_corpora,
    save_classification_outputs,
)
from safer_core.paths import layout_method_output
from safer_core.test_corpus import resolve_test_corpus


def _metadata_for_export(df: pd.DataFrame, cfg: ContrastiveConfig) -> pd.DataFrame:
    meta = df.copy()
    if "doc_id" not in meta.columns:
        meta = meta.reset_index(drop=True)
        meta["doc_id"] = [f"row_{i}" for i in range(len(meta))]
    return meta


def _encode_corpus_df(
    cfg: ContrastiveConfig,
    df: pd.DataFrame,
    text_col: str,
    checkpoint_dir: Path,
    device: str,
) -> Any:
    texts = df[text_col].astype(str).tolist()
    return encode_contrastive_texts(
        cfg,
        texts,
        checkpoint_dir=checkpoint_dir,
        device=device,
        batch_size=cfg.encode_batch_size,
    )


def run_final_classification_eval(
    cfg: ContrastiveConfig,
    checkpoint_dir: Path,
    output_root: Path,
    *,
    cv_summary: Optional[pd.DataFrame] = None,
) -> Dict[str, Path]:
    """Encode BTP + corpus test, export embeddings, LR sklearn, CSV classification."""
    layout = layout_method_output(cfg.method_name, str(output_root))
    metrics_dir = Path(layout["metrics"])
    emb_dir = Path(layout["embeddings"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    anchor = Path(__file__).resolve().parents[1]

    btp_dataset = prepare_text_dataset(cfg)
    btp_df = btp_dataset.metadata_df
    text_col = cfg.text_col
    label_col = cfg.label_col
    group_col = cfg.group_col

    X_btp = _encode_corpus_df(cfg, btp_df, text_col, checkpoint_dir, device)
    export_projected_embeddings(
        X_btp,
        _metadata_for_export(btp_df, cfg),
        emb_dir,
        "btp",
        label_col=label_col,
        group_col=group_col,
        text_col=text_col,
    )

    macros = None
    y_train_int = btp_df["label_id"].astype(int).to_numpy()
    pipe = fit_classifier_on_embeddings(X_btp, y_train_int, cfg, seed=cfg.seed)

    metrics_by_corpus: Dict[str, Mapping[str, Any]] = {}
    y_btp_macro = btp_df[label_col].astype(str).to_numpy()
    metrics_by_corpus["btp"] = evaluate_classifier_on_embeddings(pipe, X_btp, y_btp_macro, macros=macros)

    for corpus_id in cfg.test_corpora_list():
        try:
            spec = resolve_test_corpus(corpus_id, anchor=anchor)
            test_cfg = ContrastiveConfig(
                method_name=cfg.method_name,
                dataset_path=spec.data_csv,
                text_col=cfg.text_col,
                label_col=cfg.label_col,
                group_col=cfg.group_col,
                pred_ok_col=cfg.pred_ok_col,
                backbone_name=cfg.backbone_name,
                max_seq_length=cfg.max_seq_length,
                encode_batch_size=cfg.encode_batch_size,
                eval_batch_size=cfg.eval_batch_size,
                backbone_trainable=cfg.backbone_trainable,
                train_last_n_layers=cfg.train_last_n_layers,
                cache_backbone_embeddings=cfg.cache_backbone_embeddings,
                use_projector=cfg.use_projector,
                projection=cfg.projection,
                hiddim=cfg.hiddim,
                post_eval_enabled=cfg.post_eval_enabled,
                post_eval_classifier=cfg.post_eval_classifier,
                post_eval_class_weight=cfg.post_eval_class_weight,
                post_eval_oversampling=cfg.post_eval_oversampling,
            )
            test_dataset = prepare_text_dataset(test_cfg)
            test_df = test_dataset.metadata_df
            X_test = _encode_corpus_df(cfg, test_df, text_col, checkpoint_dir, device)
            export_projected_embeddings(
                X_test,
                _metadata_for_export(test_df, cfg),
                emb_dir,
                str(corpus_id),
                label_col=label_col,
                group_col=group_col,
                text_col=text_col,
            )
            y_test = test_df[label_col].astype(str).to_numpy()
            metrics_by_corpus[str(corpus_id)] = evaluate_classifier_on_embeddings(
                pipe, X_test, y_test, macros=macros
            )
        except Exception as exc:
            print(f"[{cfg.method_name}] eval corpus {corpus_id} ignorée : {exc}", flush=True)

    if cv_summary is None:
        kfold_path = metrics_dir / "kfold_summary.csv"
        cv_path = Path(layout["root"]) / "cv" / "cv_summary.csv"
        if cv_path.is_file():
            cv_summary = pd.read_csv(cv_path)
        elif kfold_path.is_file():
            cv_summary = build_cv_summary_from_kfold(pd.read_csv(kfold_path), model_name=cfg.method_name)
        else:
            cv_summary = pd.DataFrame()

    return save_classification_outputs(
        Path(layout["root"]),
        method_name=cfg.method_name,
        metrics_by_corpus=metrics_by_corpus,
        cv_summary=cv_summary,
        classifier=cfg.post_eval_classifier,
    )


def evaluate_btp_and_test(
    cfg: ContrastiveConfig,
    checkpoint_dir: Path,
    output_root: Path,
) -> Dict[str, Path]:
    """Alias compat — classification multi-corpus + 3 embeddings."""
    return run_final_classification_eval(cfg, checkpoint_dir, output_root)
