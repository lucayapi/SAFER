"""Évaluation géométrique η² / IPR sur embeddings projetés z (supervised_macro_ft)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from contrastive_methods.eval_geometry import compute_fold_ipr, evaluate_embeddings_geometry
from metrics.geometry import GEOMETRY_METRIC_KEYS, METRICS_TABLE_COLUMNS
from metrics.intra_role_preservation import IPR_COLUMNS, compute_ipr_from_geometry_rows
from safer_core.kfold_eval import save_kfold_tables
from scgm_text.dataset_text_raw import TextRawDataset
from supervised_macro_ft.embedding_cache import encode_projected_matrix
from supervised_macro_ft.model import SupervisedMacroModel

logger = logging.getLogger(__name__)

METHOD_LABEL_VAL = "Supervised macro FT (val)"
METHOD_LABEL_BTP = "Supervised macro FT (BTP)"
METHOD_LABEL_TEST = "Supervised macro FT (test)"


def geometry_keys_from_row(geom: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in GEOMETRY_METRIC_KEYS:
        val = geom.get(key, float("nan"))
        try:
            out[key] = float(val)
        except (TypeError, ValueError):
            out[key] = float("nan")
    return out


def evaluate_projected_geometry(
    z: np.ndarray,
    labels: np.ndarray,
    *,
    method: str = METHOD_LABEL_VAL,
) -> Dict[str, Any]:
    z_arr = np.asarray(z, dtype=np.float64)
    if z_arr.ndim != 2 or z_arr.shape[0] == 0:
        raise ValueError(f"embeddings projetés invalides : {getattr(z_arr, 'shape', None)}")
    return evaluate_embeddings_geometry(
        z_arr,
        labels,
        method=method,
        embedding_dim=int(z_arr.shape[1]),
    )


def encode_val_projected(
    model: SupervisedMacroModel,
    dataset: TextRawDataset,
    val_idx: np.ndarray,
    *,
    backbone_hidden: Optional[np.ndarray],
    tokenizer,
    device: torch.device,
    model_cfg: Mapping[str, Any],
    batch_size: int,
) -> np.ndarray:
    """Encode z sur le val fold (cache backbone ou forward texte)."""
    if backbone_hidden is not None:
        h_val = np.asarray(backbone_hidden, dtype=np.float32)[val_idx]
        return encode_projected_matrix(
            model,
            h_val,
            batch_size=batch_size,
            device=device,
        )
    from supervised_macro_ft.transfer import encode_texts

    meta = dataset.get_metadata_df().iloc[val_idx]
    text_col = dataset.text_col
    texts = meta[text_col].astype(str).tolist()
    max_length = int(model_cfg.get("max_seq_length", 256))
    return encode_texts(
        model,
        tokenizer,
        texts,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )


def evaluate_fold_geometry(
    z_val: np.ndarray,
    val_meta: pd.DataFrame,
    label_col: str,
    *,
    raw_emb_csv: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    """η² + IPR sur val fold (z projeté vs Qwen brut même fold)."""
    labels = val_meta[label_col].astype(str).to_numpy()
    geom_row = evaluate_projected_geometry(z_val, labels, method=METHOD_LABEL_VAL)
    geom_keys = geometry_keys_from_row(geom_row)
    ipr = compute_fold_ipr(val_meta, label_col, geom_keys, emb_csv=raw_emb_csv)
    return {**geom_keys, **ipr}


def evaluate_corpus_geometry_with_ipr(
    z: np.ndarray,
    meta_df: pd.DataFrame,
    label_col: str,
    *,
    method: str,
    raw_emb_csv: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Géométrie complète + IPR sur un corpus entier (BTP final ou test)."""
    labels = meta_df[label_col].astype(str).to_numpy()
    geom_row = evaluate_projected_geometry(z, labels, method=method)
    out = dict(geom_row)
    if raw_emb_csv and Path(str(raw_emb_csv)).is_file():
        try:
            from contrastive_methods.eval_geometry import evaluate_raw_val_geometry

            raw_geom = evaluate_raw_val_geometry(meta_df, label_col, emb_csv=raw_emb_csv)
            ipr = compute_ipr_from_geometry_rows(raw_geom, geom_row)
            out.update(ipr)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            logger.warning("IPR ignoré (%s) : %s", method, exc)
            out.update({col: float("nan") for col in IPR_COLUMNS})
    else:
        out.update({col: float("nan") for col in IPR_COLUMNS})
    return out


def save_geometry_kfold_tables(
    fold_rows: Sequence[Mapping[str, Any]],
    metrics_dir: str | Path,
    *,
    prefix: str = "kfold_geometry",
) -> None:
    """Écrit kfold_geometry_per_fold.csv et kfold_geometry_summary.csv (μ±σ)."""
    rows = [dict(row) for row in fold_rows]
    for row in rows:
        row.setdefault("method", "supervised_macro_ft")
    save_kfold_tables(list(rows), metrics_dir, prefix=prefix)


def save_geometry_metrics_csv(row: Mapping[str, Any], path: str | Path) -> Path:
    """Une ligne METRICS_TABLE_COLUMNS + IPR optionnel."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out_row: Dict[str, Any] = {}
    for col in METRICS_TABLE_COLUMNS:
        if col in row:
            out_row[col] = row[col]
    for col in IPR_COLUMNS:
        if col in row:
            out_row[col] = row[col]
    if "method" not in out_row and row.get("method"):
        out_row["method"] = row["method"]
    pd.DataFrame([out_row]).to_csv(path, index=False)
    return path


def geometry_summary_for_notebook(summary_path: str | Path) -> pd.DataFrame:
    """Charge kfold_geometry_summary et ajoute colonne method lisible."""
    path = Path(summary_path)
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "method" not in df.columns:
        df.insert(0, "method", "Supervised macro FT (CE)")
    else:
        df["method"] = df["method"].astype(str).replace(
            {"supervised_macro_ft": "Supervised macro FT (CE)"}
        )
    return df
