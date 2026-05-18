"""Évaluation géométrique SCGM sur corpus BTP / test (embeddings Qwen figés)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from metrics.geometry import build_geometry_metrics_row
from safer_core.io import save_metrics_geometry
from safer_core.paths import TEXT_ROOT, layout_method_output
from scgm_text.batch_utils import forward_features
from scgm_text.checkpoint_io import load_scgm_checkpoint
from scgm_text.dataset_text_embeddings import TextEmbeddingDataset


def project_embedding_corpus(
    checkpoint_path: str,
    data_csv: str,
    emb_csv: str,
    *,
    label_col: str = "pred_label",
    pred_ok_col: str = "pred_ok",
    group_col: str = "accident_id",
    batch_size: int = 512,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Projette un CSV + embeddings figés via best_model.pt."""
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model, checkpoint_args, _ = load_scgm_checkpoint(checkpoint_path, map_location="cpu")
    model.to(dev)
    model.eval()

    dataset = TextEmbeddingDataset(
        data_csv=data_csv,
        emb_csv=emb_csv,
        label_col=label_col,
        pred_ok_col=pred_ok_col,
        group_col=group_col,
    )
    meta = dataset.get_metadata_df()
    labels = meta[label_col].to_numpy()
    projected: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            batch_embeddings = []
            for index in range(start, end):
                embedding, _, _ = dataset[index]
                batch_embeddings.append(embedding)
            embeddings = torch.stack(batch_embeddings).to(dev)
            features = model(embeddings)
            projected.append(features.cpu().numpy())

    return np.concatenate(projected, axis=0), labels


def save_scgm_projected_corpus(
    checkpoint_path: str,
    data_csv: str,
    emb_csv: str,
    emb_dir: Path,
    *,
    stem: str = "test",
    label_col: str = "pred_label",
    pred_ok_col: str = "pred_ok",
    group_col: str = "accident_id",
    batch_size: int = 512,
) -> Dict[str, Path]:
    """Projette et persiste .npy + metadata CSV pour le notebook (PCA / UMAP)."""
    emb_dir = Path(emb_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(data_csv)
    emb_path = Path(emb_csv)
    if not data_path.is_absolute():
        data_path = TEXT_ROOT / data_path
    if not emb_path.is_absolute():
        emb_path = TEXT_ROOT / emb_path

    projected, _ = project_embedding_corpus(
        checkpoint_path,
        str(data_path),
        str(emb_path),
        label_col=label_col,
        pred_ok_col=pred_ok_col,
        group_col=group_col,
        batch_size=batch_size,
    )

    npy_name = "projected_embeddings.npy" if stem == "btp" else f"projected_embeddings_{stem}.npy"
    meta_name = "metadata_with_predictions.csv" if stem == "btp" else f"{stem}_metadata.csv"
    npy_path = emb_dir / npy_name
    meta_path = emb_dir / meta_name
    np.save(npy_path, projected.astype(np.float32))

    dataset = TextEmbeddingDataset(
        data_csv=str(data_path),
        emb_csv=str(emb_path),
        label_col=label_col,
        pred_ok_col=pred_ok_col,
        group_col=group_col,
    )
    meta = dataset.get_metadata_df()
    keep: List[str] = [c for c in ("doc_id", "accident_id", "fact_id", label_col, "sentence") if c in meta.columns]
    meta[keep].to_csv(meta_path, index=False)
    return {"projections": npy_path, "metadata": meta_path}


def evaluate_scgm_on_corpus(
    checkpoint_path: str,
    data_csv: str,
    emb_csv: str,
    *,
    corpus: str = "btp",
    metrics_dir: Optional[Path] = None,
    label_col: str = "pred_label",
    pred_ok_col: str = "pred_ok",
    group_col: str = "accident_id",
) -> Dict[str, Any]:
    projected, labels = project_embedding_corpus(
        checkpoint_path,
        data_csv,
        emb_csv,
        label_col=label_col,
        pred_ok_col=pred_ok_col,
        group_col=group_col,
    )
    row = build_geometry_metrics_row(
        projected,
        labels,
        method=f"SCGM_{corpus}",
        l2_normalize=True,
    )
    if metrics_dir is not None:
        save_metrics_geometry(row, metrics_dir, stem=f"metrics_geometry_{corpus}")
    return row


def evaluate_scgm_btp_and_test(
    checkpoint_path: str,
    output_root: str,
    *,
    data_btp: str,
    emb_btp: str,
    data_test: str,
    emb_test: str,
    label_col: str = "pred_label",
    pred_ok_col: str = "pred_ok",
    group_col: str = "accident_id",
) -> Dict[str, Path]:
    layout = layout_method_output("scgm_text", output_root)
    metrics_dir = Path(layout["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    evaluate_scgm_on_corpus(
        checkpoint_path,
        str(TEXT_ROOT / data_btp) if not Path(data_btp).is_absolute() else data_btp,
        str(TEXT_ROOT / emb_btp) if not Path(emb_btp).is_absolute() else emb_btp,
        corpus="btp",
        metrics_dir=metrics_dir,
        label_col=label_col,
        pred_ok_col=pred_ok_col,
        group_col=group_col,
    )
    paths["btp"] = metrics_dir / "metrics_geometry_btp.csv"

    test_data = TEXT_ROOT / data_test if not Path(data_test).is_absolute() else Path(data_test)
    test_emb = TEXT_ROOT / emb_test if not Path(emb_test).is_absolute() else Path(emb_test)
    if test_data.is_file() and test_emb.is_file():
        evaluate_scgm_on_corpus(
            checkpoint_path,
            str(test_data),
            str(test_emb),
            corpus="test",
            metrics_dir=metrics_dir,
            label_col=label_col,
            pred_ok_col=pred_ok_col,
            group_col=group_col,
        )
        paths["test"] = metrics_dir / "metrics_geometry_test.csv"
    return paths


def evaluate_and_save_btp_test(
    checkpoint_path: str,
    output_root: str,
    *,
    data_btp: str,
    emb_btp: str,
    data_test: str,
    emb_test: str,
    label_col: str = "pred_label",
    pred_ok_col: str = "pred_ok",
    group_col: str = "accident_id",
    save_projections: bool = True,
) -> Dict[str, Path]:
    """Métriques géométrie + sauvegarde des projections pour notebook."""
    paths = evaluate_scgm_btp_and_test(
        checkpoint_path,
        output_root,
        data_btp=data_btp,
        emb_btp=emb_btp,
        data_test=data_test,
        emb_test=emb_test,
        label_col=label_col,
        pred_ok_col=pred_ok_col,
        group_col=group_col,
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
            emb_btp,
            emb_dir,
            stem="btp",
            label_col=label_col,
            pred_ok_col=pred_ok_col,
            group_col=group_col,
        )
        paths["projections_btp"] = saved["projections"]
    test_data = TEXT_ROOT / data_test if not Path(data_test).is_absolute() else Path(data_test)
    test_emb = TEXT_ROOT / emb_test if not Path(emb_test).is_absolute() else Path(emb_test)
    if not test_data.is_file():
        print(f"[eval] Projections test ignorées : data absent → {test_data}", flush=True)
    elif not test_emb.is_file():
        print(
            f"[eval] Projections test ignorées : embeddings test absents → {test_emb}\n"
            "  Lancez : python scripts/export_test_embeddings.py",
            flush=True,
        )
    if test_data.is_file() and test_emb.is_file():
        saved_test = save_scgm_projected_corpus(
            checkpoint_path,
            str(test_data),
            str(test_emb),
            emb_dir,
            stem="test",
            label_col=label_col,
            pred_ok_col=pred_ok_col,
            group_col=group_col,
        )
        paths["projections_test"] = saved_test["projections"]
        paths["metadata_test"] = saved_test["metadata"]
    return paths
