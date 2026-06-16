"""Évaluation géométrique val (eta2_macro_balanced_perc = 100×η²) pour sélection de checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd

from contrastive_methods.config import ContrastiveConfig
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

    emb_path = resolve_btp_raw_emb_csv(emb_csv)
    if not emb_path.is_file():
        raise FileNotFoundError(f"emb_csv absent pour IPR val : {emb_path}")
    slim = val_df.drop(columns=[c for c in val_df.columns if c.startswith("dim_")], errors="ignore")
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


def _load_st_model(checkpoint_dir: Path, cfg: ContrastiveConfig):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(checkpoint_dir), trust_remote_code=True)
    if cfg.max_seq_length:
        model.max_seq_length = int(cfg.max_seq_length)
    return model


def _load_softtriple_encoder(checkpoint_dir: Path, cfg: ContrastiveConfig, device: str):
    import torch
    from contrastive_methods.losses.softtriple import HFTextEncoder

    encoder = HFTextEncoder(cfg.backbone_name, gradient_checkpointing=False).to(device)
    ckpt = checkpoint_dir / "hf_model.bin"
    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    encoder.encoder.load_state_dict(state)
    return encoder


def encode_contrastive_texts(
    cfg: ContrastiveConfig,
    texts: List[str],
    *,
    checkpoint_dir: Optional[Path] = None,
    st_model=None,
    hf_encoder=None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    normalize_embeddings: bool = METRIC_EVAL_NORMALIZE,
) -> np.ndarray:
    """Encode un corpus pour métriques géométrie (chemin unique ST / SoftTriple)."""
    bs = batch_size or cfg.encode_batch_size
    if cfg.method_name == "softtriple":
        from contrastive_methods.losses.softtriple import encode_texts_with_hf_encoder
        from contrastive_methods.st_common import get_device

        dev = device or get_device()
        encoder = hf_encoder
        if encoder is None:
            if checkpoint_dir is None:
                raise ValueError("softtriple encode : checkpoint_dir ou hf_encoder requis")
            encoder = _load_softtriple_encoder(Path(checkpoint_dir), cfg, dev)
        return encode_texts_with_hf_encoder(
            encoder,
            texts,
            batch_size=bs,
            device=dev,
            normalize_embeddings=normalize_embeddings,
            max_length=cfg.max_seq_length,
        )

    model = st_model
    if model is None:
        if checkpoint_dir is None:
            raise ValueError("encode ST : checkpoint_dir ou st_model requis")
        model = _load_st_model(Path(checkpoint_dir), cfg)
    emb = model.encode(
        texts,
        batch_size=bs,
        show_progress_bar=False,
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=True,
    )
    return np.asarray(emb)


def evaluate_st_val_geometry(
    model,
    val_df: pd.DataFrame,
    cfg: ContrastiveConfig,
    text_col: str,
) -> Dict[str, Any]:
    if hasattr(model, "eval"):
        model.eval()
    texts = val_df[text_col].astype(str).tolist()
    labels = val_df[cfg.label_col].to_numpy()
    emb = encode_contrastive_texts(cfg, texts, st_model=model, batch_size=cfg.eval_batch_size)
    return evaluate_embeddings_geometry(emb, labels, method="val")


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
    return evaluate_embeddings_geometry(emb, labels, method="val")
