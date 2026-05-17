"""Export embeddings corpus → CSV dim_* unifié."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import numpy as np
import pandas as pd

from scgm_text.dataset_text_raw import TextRawDataset

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def dim_column_names(dim: int) -> List[str]:
    return [f"dim_{i:04d}" for i in range(1, dim + 1)]


def embeddings_to_dataframe(
    doc_ids: np.ndarray,
    embeddings: np.ndarray,
) -> pd.DataFrame:
    dim = embeddings.shape[1]
    frame = pd.DataFrame({"doc_id": doc_ids.astype(np.int64)})
    cols = dim_column_names(dim)
    frame[cols] = embeddings.astype(np.float32)
    return frame


def export_st_embeddings(
    model: Union["SentenceTransformer", str, Path],
    dataset: TextRawDataset,
    dest_csv: Path,
    *,
    batch_size: int = 128,
    normalize: bool = True,
    show_progress: bool = False,
) -> Path:
    from sentence_transformers import SentenceTransformer

    if not isinstance(model, SentenceTransformer):
        model = SentenceTransformer(str(model))

    texts = dataset.metadata_df[dataset.text_col].astype(str).tolist()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    doc_ids = dataset.metadata_df["doc_id"].to_numpy()
    frame = embeddings_to_dataframe(doc_ids, np.asarray(embeddings))
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest_csv, index=False)
    return dest_csv
