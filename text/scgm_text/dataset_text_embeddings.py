"""Dataset et fusion métadonnées + embeddings pré-calculés (CSV dim_*)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset

from scgm_text.data_metadata import ID2LABEL, LABEL2ID, VALID_LABELS, load_filtered_metadata
from scgm_text.utils_io import get_dim_columns


def merge_metadata_with_embeddings(
    metadata_df: pd.DataFrame,
    emb_csv: str,
) -> Tuple[pd.DataFrame, List[str]]:
    header = pd.read_csv(emb_csv, nrows=0)
    dim_columns = get_dim_columns(header)
    emb_df = pd.read_csv(emb_csv, usecols=["doc_id"] + dim_columns)
    merged = metadata_df.merge(emb_df, on="doc_id", how="inner", validate="one_to_one")
    if len(merged) != len(metadata_df):
        n_meta = len(metadata_df)
        n_emb = len(emb_df)
        n_merged = len(merged)
        n_missing = n_meta - n_merged
        raise ValueError(
            f"Embedding merge dropped rows: metadata={n_meta}, embeddings={n_emb}, merged={n_merged} "
            f"({n_missing} doc_id sans vecteur). "
            f"Le CSV d'embeddings est probablement obsolète par rapport à {emb_csv!r}. "
            "Relancer l'export encodeur, par ex. :\n"
            "  python scripts/export_corpus_embeddings.py --corpus btp --force\n"
            "  # ou : FORCE=1 CORPUS=btp bash jobs/export_corpus_embeddings.sh"
        )
    merged.reset_index(drop=True, inplace=True)
    return merged, dim_columns


class TextEmbeddingDataset(Dataset):
    def __init__(
        self,
        data_csv: str,
        emb_csv: str,
        label_col: str = "pred_label",
        pred_ok_col: str = "pred_ok",
        group_col: str = "accident_id",
        metadata_df: Optional[pd.DataFrame] = None,
        dim_columns: Optional[Sequence[str]] = None,
    ) -> None:
        if metadata_df is None:
            metadata_df = load_filtered_metadata(
                data_csv=data_csv,
                label_col=label_col,
                pred_ok_col=pred_ok_col,
                group_col=group_col,
            )
            metadata_df, dim_columns = merge_metadata_with_embeddings(metadata_df, emb_csv)
        else:
            if dim_columns is None:
                raise ValueError("dim_columns must be provided when metadata_df is supplied.")
            metadata_df = metadata_df.copy()
            metadata_df.reset_index(drop=True, inplace=True)

        self.metadata_df = metadata_df
        self.dim_columns = list(dim_columns)
        self.label_col = label_col
        self.group_col = group_col
        self.embeddings = metadata_df[self.dim_columns].to_numpy(dtype=np.float32)
        self.label_ids = metadata_df["label_id"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = torch.from_numpy(self.embeddings[index])
        label_id = torch.tensor(self.label_ids[index], dtype=torch.long)
        selected_index = torch.tensor(index, dtype=torch.long)
        return embedding, label_id, selected_index

    def get_metadata_df(self) -> pd.DataFrame:
        return self.metadata_df.copy()

    def get_input_dim(self) -> int:
        return len(self.dim_columns)

    def get_label_distribution(self) -> Dict[str, int]:
        counts = self.metadata_df[self.label_col].value_counts()
        return {str(label): int(count) for label, count in counts.items()}

    def get_groups(self) -> np.ndarray:
        return self.metadata_df[self.group_col].astype(str).to_numpy()


def split_by_group(dataset, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    groups = dataset.get_groups()
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(dataset)), groups=groups))
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


class IndexedSubset(Subset):
    def __getitem__(self, index: int):
        embedding, label_id, _ = self.dataset[self.indices[index]]
        return embedding, label_id, torch.tensor(index, dtype=torch.long)


def build_dataloaders(
    dataset: TextEmbeddingDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        IndexedSubset(dataset, train_idx.tolist()),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


__all__ = [
    "ID2LABEL",
    "LABEL2ID",
    "VALID_LABELS",
    "TextEmbeddingDataset",
    "build_dataloaders",
    "load_filtered_metadata",
    "merge_metadata_with_embeddings",
    "split_by_group",
]
