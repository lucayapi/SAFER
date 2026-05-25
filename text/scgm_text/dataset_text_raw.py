"""Dataset texte brut (tokenisation au collate) — pipeline SCGM end2end."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset

from scgm_text.data_metadata import load_filtered_metadata, resolve_text_column


def split_by_group(dataset: Dataset, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    groups = dataset.get_groups()  # type: ignore[attr-defined]
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(dataset)), groups=groups))
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


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
                text_col=text_col,
            )
        else:
            metadata_df = metadata_df.copy()
            metadata_df.reset_index(drop=True, inplace=True)
            if "row_id" not in metadata_df.columns:
                metadata_df["row_id"] = np.arange(len(metadata_df), dtype=np.int64)

        self.metadata_df = metadata_df
        self.label_col = label_col
        self.group_col = group_col
        self.text_col = resolve_text_column(metadata_df, text_col)
        self.texts = metadata_df[self.text_col].astype(str).tolist()
        self.label_ids = metadata_df["label_id"].to_numpy(dtype=np.int64)
        self.groups = metadata_df[group_col].astype(str).tolist()
        self.row_ids = metadata_df["row_id"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return {
            "text": self.texts[index],
            "label": int(self.label_ids[index]),
            "group": self.groups[index],
            "row_id": int(self.row_ids[index]),
            "index": index,
        }

    def get_metadata_df(self) -> pd.DataFrame:
        return self.metadata_df.copy()

    def get_label_distribution(self) -> Dict[str, int]:
        counts = self.metadata_df[self.label_col].value_counts()
        return {str(label): int(count) for label, count in counts.items()}

    def get_groups(self) -> np.ndarray:
        return self.metadata_df[self.group_col].astype(str).to_numpy()


class IndexedTextSubset(Subset):
    def __getitem__(self, index: int) -> Dict[str, object]:
        item = self.dataset[self.indices[index]]
        item = dict(item)
        item["index"] = index
        return item


def build_text_dataloaders(
    dataset: TextRawDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
    collate_fn,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        IndexedTextSubset(dataset, train_idx.tolist()),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
