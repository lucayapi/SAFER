"""Évaluation géométrique val (eta2_macro_balanced_perc = 100×η²) pour sélection de checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.encoder_model import load_contrastive_encoder_from_checkpoint
from metrics.embedding_dims import QWEN3_EMBEDDING_06B_DIM
from metrics.geometry import PRIMARY_SELECTION_METRIC, build_geometry_metrics_row

SELECTION_METRIC_DEFAULT = PRIMARY_SELECTION_METRIC

# Normalisation L2 à l'encode pour aligner val / BTP / test (η² sur distances euclidiennes²).
METRIC_EVAL_NORMALIZE = True

DEFAULT_BTP_RAW_EMB_CSV = "embeddings/Qwen3-Embedding-0.6B_btp.csv"


def resolve_btp_raw_emb_csv(emb_csv: Optional[Union[str, Path]] = None) -> Path:
    from safer_core.paths import TEXT_ROOT

    if emb_csv is None:
        return TEXT_ROOT / DEFAULT_BTP_RAW_EMB_CSV
    path = Path(emb_csv)
    if path.is_file():
        return path
    rooted = TEXT_ROOT / path
    if rooted.is_file():
        return rooted
    return path


def evaluate_raw_val_geometry(
    val_df: pd.DataFrame,
    label_col: str,
    *,
    emb_csv: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Géométrie η² / W_r sur embeddings Qwen pré-calculés (même val fold)."""
    from scgm_text.dataset_text_embeddings import merge_metadata_with_embeddings
    from scgm_text.utils_io import create_doc_id_if_missing

    emb_path = resolve_btp_raw_emb_csv(emb_csv)
    if not emb_path.is_file():
        raise FileNotFoundError(f"emb_csv absent pour IPR val : {emb_path}")
    slim = val_df.drop(columns=[c for c in val_df.columns if c.startswith("dim_")], errors="ignore")
    slim = create_doc_id_if_missing(slim)
    merged, dim_cols = merge_metadata_with_embeddings(slim, str(emb_path))
    if merged.empty:
        raise ValueError("aucune ligne après fusion val / embeddings bruts")
    raw = merged[dim_cols].to_numpy(dtype=np.float64)
    labels = merged[label_col].to_numpy()
    return evaluate_embeddings_geometry(
        raw,
        labels,
        method="raw_val",
        embedding_dim=len(dim_cols),
    )


def compute_fold_ipr(
    val_df: pd.DataFrame,
    label_col: str,
    method_geom: Mapping[str, Any],
    *,
    emb_csv: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    """IPR sur le val fold (brut Qwen vs géométrie méthode) ; NaN si emb absent."""
    from metrics.intra_role_preservation import IPR_COLUMNS, compute_ipr_from_geometry_rows

    try:
        raw_geom = evaluate_raw_val_geometry(val_df, label_col, emb_csv=emb_csv)
        return compute_ipr_from_geometry_rows(raw_geom, method_geom)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[IPR] fold ignoré : {exc}", flush=True)
        return {col: float("nan") for col in IPR_COLUMNS}


def selection_score(row: Dict[str, Any], metric: str = SELECTION_METRIC_DEFAULT) -> float:
    value = float(row.get(metric, float("nan")))
    if not np.isfinite(value):
        return float("-inf")
    return value


def evaluate_embeddings_geometry(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    method: str = "val",
    embedding_dim: int = QWEN3_EMBEDDING_06B_DIM,
) -> Dict[str, Any]:
    return build_geometry_metrics_row(
        embeddings,
        labels,
        method=method,
        l2_normalize=True,
        embedding_dim=embedding_dim,
    )


def _load_encoder_for_eval(
    checkpoint_dir: Path,
    cfg: ContrastiveConfig,
    device: str,
):
    return load_contrastive_encoder_from_checkpoint(cfg, checkpoint_dir, device)


def encode_contrastive_texts(
    cfg: ContrastiveConfig,
    texts: List[str],
    *,
    checkpoint_dir: Optional[Path] = None,
    hf_encoder=None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    normalize_embeddings: bool = METRIC_EVAL_NORMALIZE,
    **_,
) -> np.ndarray:
    """Encode un corpus via l'encodeur HF unifié (checkpoint ou instance fournie)."""
    from contrastive_methods.hf_training_common import encode_texts, get_device

    dev = device or get_device()
    encoder = hf_encoder
    if encoder is None:
        if checkpoint_dir is None:
            raise ValueError("encode contrastif : checkpoint_dir ou hf_encoder requis")
        encoder = _load_encoder_for_eval(Path(checkpoint_dir), cfg, dev)
    return encode_texts(
        encoder,
        texts,
        cfg,
        dev,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )


def evaluate_hf_val_geometry(
    encoder,
    val_df: pd.DataFrame,
    cfg: ContrastiveConfig,
    text_col: str,
    device: str,
) -> Dict[str, Any]:
    texts = val_df[text_col].astype(str).tolist()
    labels = val_df[cfg.label_col].to_numpy()
    emb = encode_contrastive_texts(
        cfg,
        texts,
        hf_encoder=encoder,
        batch_size=cfg.eval_batch_size,
        device=device,
    )
    dim = getattr(encoder, "embedding_dim", emb.shape[1] if emb.ndim == 2 else QWEN3_EMBEDDING_06B_DIM)
    return evaluate_embeddings_geometry(emb, labels, method="val", embedding_dim=dim)
