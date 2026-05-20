"""Encodage corpus cible via checkpoint SCGM ou SoftTriple."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

from contrastive_methods.config import ContrastiveConfig, load_contrastive_config
from contrastive_methods.eval_geometry import encode_contrastive_texts
from scgm_text.eval_corpus import project_embedding_corpus

MethodName = Literal["scgm_text", "softtriple"]


def load_target_metadata(
    data_csv: str,
    *,
    text_col: str = "sentence",
) -> pd.DataFrame:
    """Charge le CSV cible (toutes les lignes, sans filtre pred_ok)."""
    df = pd.read_csv(data_csv)
    if text_col not in df.columns:
        raise ValueError(f"Colonne {text_col!r} absente de {data_csv}")
    return df.reset_index(drop=True)


def encode_target_corpus(
    method: MethodName,
    checkpoint: str,
    data_csv: str,
    emb_csv: Optional[str],
    *,
    text_col: str = "sentence",
    label_col: str = "pred_label",
    pred_ok_col: str = "pred_ok",
    group_col: str = "accident_id",
    batch_size: int = 512,
    max_seq_length: int = 256,
    device: str = "cuda",
    contrastive_config: Optional[Path] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Retourne ``z`` L2-normalisé (N, d) et métadonnées alignées ligne à ligne.
    """
    meta = load_target_metadata(data_csv, text_col=text_col)
    if method == "scgm_text":
        if not emb_csv:
            raise ValueError("scgm_text requiert --emb-csv (embeddings Qwen figés)")
        z, _labels = project_embedding_corpus(
            checkpoint,
            data_csv,
            emb_csv,
            label_col=label_col,
            pred_ok_col=pred_ok_col,
            group_col=group_col,
            text_col=text_col,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            device=device,
        )
        return np.asarray(z, dtype=np.float64), meta

    if method == "softtriple":
        ckpt_dir = Path(checkpoint)
        if ckpt_dir.is_file():
            ckpt_dir = ckpt_dir.parent
        cfg_path = contrastive_config
        if cfg_path is None:
            for candidate in (
                ckpt_dir.parent.parent / "configs" / "softtriple.yaml",
                Path(__file__).resolve().parent.parent / "configs" / "softtriple.yaml",
            ):
                if candidate.is_file():
                    cfg_path = candidate
                    break
        if cfg_path is None or not Path(cfg_path).is_file():
            cfg = ContrastiveConfig(method_name="softtriple", dataset_path=Path(data_csv))
        else:
            cfg = load_contrastive_config(cfg_path)
        texts = meta[text_col].astype(str).tolist()
        z = encode_contrastive_texts(
            cfg,
            texts,
            checkpoint_dir=ckpt_dir,
            batch_size=batch_size,
            device=device,
        )
        return np.asarray(z, dtype=np.float64), meta

    raise ValueError(f"Méthode non supportée : {method}")
