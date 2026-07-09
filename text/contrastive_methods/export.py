"""Export embeddings corpus → CSV dim_* unifié."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.encoder_model import build_contrastive_encoder, load_contrastive_encoder_from_checkpoint
from contrastive_methods.hf_training_common import encode_texts, get_device
from scgm_text.dataset_text_raw import TextRawDataset


def dim_column_names(dim: int) -> list[str]:
    return [f"dim_{i:04d}" for i in range(1, dim + 1)]


def embeddings_to_dataframe(
    doc_ids: np.ndarray,
    embeddings: np.ndarray,
) -> pd.DataFrame:
    dim = embeddings.shape[1]
    cols = dim_column_names(dim)
    ids_df = pd.DataFrame({"doc_id": doc_ids.astype(np.int64)})
    emb_df = pd.DataFrame(embeddings.astype(np.float32), columns=cols)
    return pd.concat([ids_df, emb_df], axis=1)


def export_text_embeddings(
    cfg: ContrastiveConfig,
    dataset: TextRawDataset,
    dest_csv: Path,
    *,
    checkpoint_dir: Optional[Path] = None,
    batch_size: Optional[int] = None,
    show_progress: bool = False,
) -> Path:
    """Encode un corpus via ``ContrastiveEncoder`` (backbone brut ou checkpoint contrastif)."""
    device = get_device()
    if checkpoint_dir is not None:
        encoder = load_contrastive_encoder_from_checkpoint(cfg, checkpoint_dir, device)
    else:
        encoder = build_contrastive_encoder(cfg).to(device)
    texts = dataset.metadata_df[dataset.text_col].astype(str).tolist()
    if show_progress:
        print(f"[export] encodage de {len(texts)} phrases…", flush=True)
    embeddings = encode_texts(
        encoder,
        texts,
        cfg,
        device,
        batch_size=batch_size or cfg.encode_batch_size,
    )
    doc_ids = dataset.metadata_df["doc_id"].to_numpy()
    frame = embeddings_to_dataframe(doc_ids, np.asarray(embeddings))
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest_csv, index=False)
    return dest_csv
