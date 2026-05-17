"""Dataset texte brut (tokenisation au collate)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from scgm_text.dataset_text_embeddings import (
    load_filtered_metadata,
    split_by_group,
)


def resolve_text_column(metadata_df: pd.DataFrame, text_col: Optional[str] = None) -> str:
    if text_col and text_col in metadata_df.columns:
        return text_col
    for candidate in ("sentence", "accident_summary", "text"):
        if candidate in metadata_df.columns:
            return candidate
    raise ValueError(
        "Colonne texte introuvable. Fournir --text_col ou ajouter sentence / accident_summary."
    )


class TextRawDataset(Dataset):
    def __init__(
        self,
        data_csv: str,
        label_col: str = "pred_label",
        pred_ok_col: str = "pred_ok",
        group_col: str = "accident_id",
        text_col: Optional[str] = None,
        metadata_df: Optional[pd.DataFrame] = None,
    ) -> None:
        if metadata_df is None:
            metadata_df = load_filtered_metadata(
                data_csv=data_csv,
                label_col=label_col,
                pred_ok_col=pred_ok_col,
                group_col=group_col,
            )
        else:
            metadata_df = metadata_df.copy()
            metadata_df.reset_index(drop=True, inplace=True)

        self.metadata_df = metadata_df
        self.label_col = label_col
        self.group_col = group_col
        self.text_col = resolve_text_column(metadata_df, text_col)
        self.texts = metadata_df[self.text_col].astype(str).tolist()
        self.label_ids = metadata_df["label_id"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        text = self.texts[index]
        label_id = torch.tensor(self.label_ids[index], dtype=torch.long)
        selected_index = torch.tensor(index, dtype=torch.long)
        return text, label_id, selected_index

    def get_metadata_df(self) -> pd.DataFrame:
        return self.metadata_df.copy()

    def get_label_distribution(self) -> Dict[str, int]:
        counts = self.metadata_df[self.label_col].value_counts()
        return {str(label): int(count) for label, count in counts.items()}

    def get_groups(self) -> np.ndarray:
        return self.metadata_df[self.group_col].astype(str).to_numpy()

    def get_input_dim(self) -> int:
        raise RuntimeError("get_input_dim() non défini avant chargement du backbone.")


class IndexedTextSubset(Subset):
    def __getitem__(self, index: int):
        text, label_id, _ = self.dataset[self.indices[index]]
        return text, label_id, torch.tensor(index, dtype=torch.long)


def build_text_dataloaders(
    dataset: TextRawDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
    collate_fn,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        IndexedTextSubset(dataset, train_idx.tolist()),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def split_raw_by_group(dataset: TextRawDataset, val_ratio: float, seed: int):
    return split_by_group(dataset, val_ratio, seed)  # type: ignore[arg-type]
