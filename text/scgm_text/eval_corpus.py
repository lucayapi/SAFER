"""Évaluation classification SCGM end2end (BTP + corpus OOD)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from safer_core.classification_eval import (
    build_cv_summary_from_kfold,
    export_projected_embeddings,
    fit_logistic_on_embeddings,
    evaluate_classifier_on_embeddings,
    resolve_test_corpora,
    save_classification_outputs,
)
from safer_core.paths import TEXT_ROOT, layout_method_output
from safer_core.test_corpus import resolve_test_corpus
from scgm_text.dataset_text_raw import TextRawDataset


def _resolve_data_path(data_csv: str) -> Path:
    p = Path(data_csv)
    return p if p.is_absolute() else TEXT_ROOT / p


def project_embedding_corpus(
    checkpoint_path: str,
    data_csv: str,
    *,
    label_col: str = "pred_label",
    pred_ok_col: str = "pred_ok",
    group_col: str = "accident_id",
    text_col: Optional[str] = None,
    batch_size: int = 32,
    max_seq_length: int = 256,
    device: str = "cuda",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Projette un corpus via best_model.pt (end2end: tokenise et encode)."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from scgm_text.batch_utils import batch_to_device, forward_features
    from scgm_text.checkpoint_io import load_scgm_checkpoint
    from scgm_text.collate import make_text_collate_fn

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model, checkpoint_args, _ = load_scgm_checkpoint(checkpoint_path, map_location="cpu")
    model.to(dev)
    model.eval()

    data_path = _resolve_data_path(data_csv)
    dataset = TextRawDataset(
        data_csv=str(data_path),
        label_col=label_col,
        pred_ok_col=pred_ok_col,
        group_col=group_col,
        text_col=text_col or "sentence",
    )
    backbone = (
        checkpoint_args.get("backbone_model_name_or_path")
        or checkpoint_args.get("backbone_name")
        or "Qwen/Qwen3-Embedding-0.6B"
    )
    tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
    collate_fn = make_text_collate_fn(tokenizer, max_seq_length)
    loader = DataLoader(dataset, batch_size=min(batch_size, 32), shuffle=False, collate_fn=collate_fn)
    meta = dataset.get_metadata_df()
    projected: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, dev)
            features = forward_features(model, batch)
            projected.append(features.cpu().numpy())
    emb = np.concatenate(projected, axis=0)
    return emb, meta


def save_scgm_projected_corpus(
    checkpoint_path: str,
    data_csv: str,
    emb_dir: Path,
    *,
    stem: str = "test",
    label_col: str = "pred_label",
    pred_ok_col: str = "pred_ok",
    group_col: str = "accident_id",
    text_col: Optional[str] = None,
    batch_size: int = 32,
    max_seq_length: int = 256,
) -> Dict[str, Path]:
    """Compat notebooks — export projected_<stem>.npy + metadata."""
    projected, meta = project_embedding_corpus(
        checkpoint_path,
        data_csv,
        label_col=label_col,
        pred_ok_col=pred_ok_col,
        group_col=group_col,
        text_col=text_col,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
    )
    npy, csv = export_projected_embeddings(
        projected,
        meta,
        emb_dir,
        stem,
        label_col=label_col,
        group_col=group_col,
        text_col=text_col or "sentence",
    )
    return {"projections": npy, "metadata": csv}


def run_fold_classification_eval(
    checkpoint_path: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    data_csv: str,
    label_col: str = "pred_label",
    text_col: str = "sentence",
    seed: int = 42,
) -> Dict[str, float]:
    """LR sklearn sur embeddings projetés SCGM (train fold → val fold)."""
    from scgm_text.dataset_text_embeddings import LABEL2ID

    X_all, meta = project_embedding_corpus(
        checkpoint_path, data_csv, label_col=label_col, text_col=text_col
    )
    idx_to_pos = {int(i): p for p, i in enumerate(meta.index)}
    tr_pos = [idx_to_pos[int(i)] for i in train_idx if int(i) in idx_to_pos]
    va_pos = [idx_to_pos[int(i)] for i in val_idx if int(i) in idx_to_pos]
    if not tr_pos or not va_pos:
        return {}
    X_tr = X_all[tr_pos]
    X_va = X_all[va_pos]
    train_meta = meta.iloc[tr_pos]
    val_meta = meta.iloc[va_pos]
    if "label_id" in train_meta.columns:
        y_train = train_meta["label_id"].astype(int).to_numpy()
    else:
        y_train = train_meta[label_col].astype(str).map(LABEL2ID).astype(int).to_numpy()
    y_val = val_meta[label_col].astype(str).to_numpy()
    pipe = fit_logistic_on_embeddings(X_tr, y_train, seed=seed)
    metrics = evaluate_classifier_on_embeddings(pipe, X_va, y_val)
    return {f"val_{k}": float(v) for k, v in metrics.items()}


def run_final_classification_eval(
    checkpoint_path: str,
    output_root: str,
    *,
    data_btp: str,
    test_corpora: Optional[List[str]] = None,
    label_col: str = "pred_label",
    pred_ok_col: str = "pred_ok",
    group_col: str = "accident_id",
    text_col: Optional[str] = None,
    batch_size: int = 32,
    max_seq_length: int = 256,
    cfg_for_corpora: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Path]:
    layout = layout_method_output("scgm_text", output_root)
    metrics_dir = Path(layout["metrics"])
    emb_dir = Path(layout["embeddings"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)
    eff_text = text_col or "sentence"
    corpora = test_corpora or resolve_test_corpora(cfg_for_corpora or {})

    X_btp, btp_meta = project_embedding_corpus(
        checkpoint_path,
        data_btp,
        label_col=label_col,
        pred_ok_col=pred_ok_col,
        group_col=group_col,
        text_col=eff_text,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
    )
    export_projected_embeddings(
        X_btp, btp_meta, emb_dir, "btp", label_col=label_col, group_col=group_col, text_col=eff_text
    )

    if "label_id" not in btp_meta.columns:
        from scgm_text.dataset_text_embeddings import LABEL2ID

        y_train_int = btp_meta[label_col].astype(str).map(LABEL2ID).astype(int).to_numpy()
    else:
        y_train_int = btp_meta["label_id"].astype(int).to_numpy()
    pipe = fit_logistic_on_embeddings(X_btp, y_train_int)
    y_btp = btp_meta[label_col].astype(str).to_numpy()
    metrics_by_corpus: Dict[str, Mapping[str, Any]] = {
        "btp": evaluate_classifier_on_embeddings(pipe, X_btp, y_btp),
    }

    for corpus_id in corpora:
        try:
            spec = resolve_test_corpus(corpus_id)
            X_test, test_meta = project_embedding_corpus(
                checkpoint_path,
                str(spec.data_csv),
                label_col=label_col,
                pred_ok_col=pred_ok_col,
                group_col=group_col,
                text_col=eff_text,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
            )
            export_projected_embeddings(
                X_test,
                test_meta,
                emb_dir,
                str(corpus_id),
                label_col=label_col,
                group_col=group_col,
                text_col=eff_text,
            )
            y_test = test_meta[label_col].astype(str).to_numpy()
            metrics_by_corpus[str(corpus_id)] = evaluate_classifier_on_embeddings(pipe, X_test, y_test)
        except Exception as exc:
            print(f"[scgm] eval corpus {corpus_id} ignorée : {exc}", flush=True)

    kfold_path = metrics_dir / "kfold_summary.csv"
    cv_path = Path(layout["root"]) / "cv" / "cv_summary.csv"
    if cv_path.is_file():
        cv_summary = pd.read_csv(cv_path)
    elif kfold_path.is_file():
        cv_summary = build_cv_summary_from_kfold(pd.read_csv(kfold_path), model_name="scgm_text")
    else:
        cv_summary = pd.DataFrame()

    return save_classification_outputs(
        Path(layout["root"]),
        method_name="scgm_text",
        metrics_by_corpus=metrics_by_corpus,
        cv_summary=cv_summary,
    )


def evaluate_and_save_btp_test(
    checkpoint_path: str,
    output_root: str,
    *,
    data_btp: str,
    data_test: str,
    test_corpus_id: Optional[str] = None,
    test_corpora: Optional[List[str]] = None,
    label_col: str = "pred_label",
    pred_ok_col: str = "pred_ok",
    group_col: str = "accident_id",
    text_col: Optional[str] = None,
    save_projections: bool = True,
    **kwargs: Any,
) -> Dict[str, Path]:
    """Classification multi-corpus + embeddings projetés (compat run_post_train_eval)."""
    corpora = test_corpora
    if corpora is None and test_corpus_id:
        corpora = [test_corpus_id]
    return run_final_classification_eval(
        checkpoint_path,
        output_root,
        data_btp=data_btp,
        test_corpora=corpora,
        label_col=label_col,
        pred_ok_col=pred_ok_col,
        group_col=group_col,
        text_col=text_col,
    )
