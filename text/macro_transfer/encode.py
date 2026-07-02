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
    from scgm_text.utils_io import create_doc_id_if_missing

    df = pd.read_csv(data_csv)
    if text_col not in df.columns:
        raise ValueError(f"Colonne {text_col!r} absente de {data_csv}")
    return create_doc_id_if_missing(df.reset_index(drop=True))


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
    encode_batch_size: Optional[int] = None,
    max_seq_length: int = 256,
    device: str = "cuda",
    contrastive_config: Optional[Path] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Retourne ``z`` L2-normalisé (N, d) et métadonnées alignées ligne à ligne.
    """
    if method == "scgm_text":
        if not emb_csv:
            raise ValueError("scgm_text requiert --emb-csv (embeddings Qwen figés)")
        from scgm_text.checkpoint_io import load_scgm_checkpoint
        from scgm_text.eval_corpus import project_embedding_corpus

        _, checkpoint_args, _ = load_scgm_checkpoint(checkpoint, map_location="cpu")
        input_mode = checkpoint_args.get("input_mode", "precomputed_embeddings")

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
        z = np.asarray(z, dtype=np.float64)

        if input_mode == "text":
            from scgm_text.dataset_text_raw import TextRawDataset

            dataset = TextRawDataset(
                data_csv=data_csv,
                label_col=label_col,
                pred_ok_col=pred_ok_col,
                group_col=group_col,
                text_col=text_col or "sentence",
            )
        else:
            from scgm_text.dataset_text_embeddings import TextEmbeddingDataset

            dataset = TextEmbeddingDataset(
                data_csv=data_csv,
                emb_csv=emb_csv,
                label_col=label_col,
                pred_ok_col=pred_ok_col,
                group_col=group_col,
            )
        meta = dataset.get_metadata_df().reset_index(drop=True)
        if len(z) != len(meta):
            raise ValueError(
                f"Alignement SCGM : {len(z)} projections vs {len(meta)} lignes métadonnées "
                f"(filtre pred_ok/labels et fusion embeddings)."
            )
        return z, meta

    meta = load_target_metadata(data_csv, text_col=text_col)

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
        hf_bs = encode_batch_size if encode_batch_size is not None else cfg.encode_batch_size
        if str(device).startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
        z = encode_contrastive_texts(
            cfg,
            texts,
            checkpoint_dir=ckpt_dir,
            batch_size=int(hf_bs),
            device=device,
        )
        return np.asarray(z, dtype=np.float64), meta

    raise ValueError(f"Méthode non supportée : {method}")
