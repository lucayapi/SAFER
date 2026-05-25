"""Évaluation géométrique SCGM end2end sur corpus BTP / test (texte tokenisé)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from metrics.embedding_dims import SCGM_DEFAULT_HIDIM
from metrics.geometry import build_geometry_metrics_row
from safer_core.io import save_metrics_geometry
from safer_core.paths import TEXT_ROOT, layout_method_output
from scgm_text.batch_utils import batch_to_device, forward_features
from scgm_text.checkpoint_io import load_scgm_checkpoint
from scgm_text.collate import make_text_collate_fn
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
) -> tuple[np.ndarray, np.ndarray]:
    """Projette un corpus via best_model.pt (end2end: tokenise et encode)."""
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
    from transformers import AutoTokenizer

    backbone = (
        checkpoint_args.get("backbone_model_name_or_path")
        or checkpoint_args.get("backbone_name")
        or "Qwen/Qwen3-Embedding-0.6B"
    )
    tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
    collate_fn = make_text_collate_fn(tokenizer, max_seq_length)
    eff_batch = min(batch_size, 32)
    loader = DataLoader(
        dataset,
        batch_size=eff_batch,
        shuffle=False,
        collate_fn=collate_fn,
    )
    meta = dataset.get_metadata_df()
    labels = meta[label_col].to_numpy()
    projected: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, dev)
            features = forward_features(model, batch)
            projected.append(features.cpu().numpy())
    return np.concatenate(projected, axis=0), labels


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
    emb_dir = Path(emb_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)
    data_path = _resolve_data_path(data_csv)
    eff_text_col = text_col or "sentence"

    projected, _ = project_embedding_corpus(
        checkpoint_path,
        str(data_path),
        label_col=label_col,
        pred_ok_col=pred_ok_col,
        group_col=group_col,
        text_col=eff_text_col,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
    )

    npy_name = "projected_embeddings.npy" if stem == "btp" else f"projected_embeddings_{stem}.npy"
    meta_name = "metadata_with_predictions.csv" if stem == "btp" else f"{stem}_metadata.csv"
    npy_path = emb_dir / npy_name
    meta_path = emb_dir / meta_name
    np.save(npy_path, projected.astype(np.float32))

    dataset = TextRawDataset(
        data_csv=str(data_path),
        label_col=label_col,
        pred_ok_col=pred_ok_col,
        group_col=group_col,
        text_col=eff_text_col,
    )
    meta = dataset.get_metadata_df()
    keep: List[str] = [
        c for c in ("doc_id", "accident_id", "fact_id", label_col, "sentence", "row_id") if c in meta.columns
    ]
    meta[keep].to_csv(meta_path, index=False)
    return {"projections": npy_path, "metadata": meta_path}


def evaluate_scgm_on_corpus(
    checkpoint_path: str,
    data_csv: str,
    *,
    corpus: str = "btp",
    metrics_dir: Optional[Path] = None,
    label_col: str = "pred_label",
    pred_ok_col: str = "pred_ok",
    group_col: str = "accident_id",
    text_col: Optional[str] = None,
) -> Dict[str, Any]:
    projected, labels = project_embedding_corpus(
        checkpoint_path,
        data_csv,
        label_col=label_col,
        pred_ok_col=pred_ok_col,
        group_col=group_col,
        text_col=text_col,
    )
    d = int(projected.shape[1]) if projected.ndim == 2 else SCGM_DEFAULT_HIDIM
    row = build_geometry_metrics_row(
        projected,
        labels,
        method=f"SCGM_{corpus}",
        l2_normalize=True,
        embedding_dim=d,
    )
    if metrics_dir is not None:
        save_metrics_geometry(row, metrics_dir, stem=f"metrics_geometry_{corpus}")
    return row


def evaluate_scgm_btp_and_test(
    checkpoint_path: str,
    output_root: str,
    *,
    data_btp: str,
    data_test: str,
    test_corpus_id: Optional[str] = None,
    label_col: str = "pred_label",
    pred_ok_col: str = "pred_ok",
    group_col: str = "accident_id",
    text_col: Optional[str] = None,
) -> Dict[str, Path]:
    from safer_core.test_corpus import method_test_results_dir

    layout = layout_method_output("scgm_text", output_root)
    metrics_dir = Path(layout["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    btp_path = _resolve_data_path(data_btp)
    evaluate_scgm_on_corpus(
        checkpoint_path,
        str(btp_path),
        corpus="btp",
        metrics_dir=metrics_dir,
        label_col=label_col,
        pred_ok_col=pred_ok_col,
        group_col=group_col,
        text_col=text_col,
    )
    paths["btp"] = metrics_dir / "metrics_geometry_btp.csv"

    test_data = _resolve_data_path(data_test)
    if test_data.is_file():
        test_metrics_dir = method_test_results_dir("scgm_text", test_corpus_id) / "metrics"
        test_metrics_dir.mkdir(parents=True, exist_ok=True)
        evaluate_scgm_on_corpus(
            checkpoint_path,
            str(test_data),
            corpus="test",
            metrics_dir=test_metrics_dir,
            label_col=label_col,
            pred_ok_col=pred_ok_col,
            group_col=group_col,
            text_col=text_col,
        )
        paths["test"] = test_metrics_dir / "metrics_geometry_test.csv"
    return paths


def evaluate_and_save_btp_test(
    checkpoint_path: str,
    output_root: str,
    *,
    data_btp: str,
    data_test: str,
    test_corpus_id: Optional[str] = None,
    label_col: str = "pred_label",
    pred_ok_col: str = "pred_ok",
    group_col: str = "accident_id",
    text_col: Optional[str] = None,
    save_projections: bool = True,
    **kwargs: Any,
) -> Dict[str, Path]:
    """Métriques géométrie + sauvegarde des projections SCGM (end2end)."""
    from safer_core.test_corpus import method_test_results_dir

    paths = evaluate_scgm_btp_and_test(
        checkpoint_path,
        output_root,
        data_btp=data_btp,
        data_test=data_test,
        test_corpus_id=test_corpus_id,
        label_col=label_col,
        pred_ok_col=pred_ok_col,
        group_col=group_col,
        text_col=text_col,
    )
    if not save_projections:
        return paths
    layout = layout_method_output("scgm_text", output_root)
    emb_dir = Path(layout["embeddings"])
    btp_npy = emb_dir / "projected_embeddings.npy"
    if not btp_npy.is_file():
        saved = save_scgm_projected_corpus(
            checkpoint_path,
            data_btp,
            emb_dir,
            stem="btp",
            label_col=label_col,
            pred_ok_col=pred_ok_col,
            group_col=group_col,
            text_col=text_col,
        )
        paths["projections_btp"] = saved["projections"]
    test_data = _resolve_data_path(data_test)
    if test_data.is_file():
        test_emb_dir = method_test_results_dir("scgm_text", test_corpus_id) / "embeddings"
        test_emb_dir.mkdir(parents=True, exist_ok=True)
        saved_test = save_scgm_projected_corpus(
            checkpoint_path,
            str(test_data),
            test_emb_dir,
            stem="test",
            label_col=label_col,
            pred_ok_col=pred_ok_col,
            group_col=group_col,
            text_col=text_col,
        )
        paths["projections_test"] = saved_test["projections"]
        paths["metadata_test"] = saved_test["metadata"]
    return paths
