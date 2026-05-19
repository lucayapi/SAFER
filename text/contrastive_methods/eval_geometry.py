"""Évaluation géométrique val (eta2_macro_balanced_perc = 100×η²) pour sélection de checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from contrastive_methods.config import ContrastiveConfig
from metrics.geometry import PRIMARY_SELECTION_METRIC, build_geometry_metrics_row

SELECTION_METRIC_DEFAULT = PRIMARY_SELECTION_METRIC

# Normalisation L2 à l'encode pour aligner val / BTP / test (η² sur distances euclidiennes²).
METRIC_EVAL_NORMALIZE = True


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
) -> Dict[str, Any]:
    return build_geometry_metrics_row(
        embeddings,
        labels,
        method=method,
        l2_normalize=True,
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
