from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from scgm_text.dataset_text_embeddings import LABEL2ID, merge_metadata_with_embeddings
from scgm_text.utils_io import create_doc_id_if_missing, parse_bool_column

METADATA_COLUMNS = [
    "doc_id",
    "accident_id",
    "fact_id",
    "sentence",
    "accident_summary",
    "pred_label",
    "pred_subtype",
    "pred_confidence",
    "division",
    "equipment_involved",
    "company_code",
]


def load_target_metadata(
    data_csv: str,
    filter_pred_ok: bool = False,
    pred_ok_col: str = "pred_ok",
) -> pd.DataFrame:
    data_df = pd.read_csv(data_csv)
    data_df = create_doc_id_if_missing(data_df)
    if filter_pred_ok:
        if pred_ok_col not in data_df.columns:
            raise ValueError(f"Missing column {pred_ok_col} while filter_pred_ok=True")
        ok_mask = parse_bool_column(data_df[pred_ok_col])
        data_df = data_df.loc[ok_mask].copy()
    data_df.reset_index(drop=True, inplace=True)
    return data_df


class MALTTargetDataset(Dataset):
    def __init__(
        self,
        data_csv: str,
        emb_csv: str,
        filter_pred_ok: bool = False,
        pred_ok_col: str = "pred_ok",
        metadata_df: Optional[pd.DataFrame] = None,
        dim_columns: Optional[Sequence[str]] = None,
        expected_input_dim: Optional[int] = None,
    ) -> None:
        if metadata_df is None:
            metadata_df = load_target_metadata(
                data_csv=data_csv,
                filter_pred_ok=filter_pred_ok,
                pred_ok_col=pred_ok_col,
            )
            metadata_df, dim_columns = merge_metadata_with_embeddings(metadata_df, emb_csv)
        else:
            if dim_columns is None:
                raise ValueError("dim_columns must be provided when metadata_df is supplied.")
            metadata_df = metadata_df.copy()
            metadata_df.reset_index(drop=True, inplace=True)

        self.metadata_df = metadata_df
        self.dim_columns = list(dim_columns)
        self.embeddings = metadata_df[self.dim_columns].to_numpy(dtype=np.float32)
        if expected_input_dim is not None and len(self.dim_columns) != expected_input_dim:
            raise ValueError(
                f"Target embedding dim {len(self.dim_columns)} != source input_dim {expected_input_dim}"
            )

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = torch.from_numpy(self.embeddings[index])
        selected_index = torch.tensor(index, dtype=torch.long)
        return embedding, selected_index

    def get_metadata_df(self) -> pd.DataFrame:
        return self.metadata_df.copy()

    def get_input_dim(self) -> int:
        return len(self.dim_columns)

    def get_diagnostic_label_ids(self) -> Optional[np.ndarray]:
        if "pred_label" not in self.metadata_df.columns:
            return None
        labels = self.metadata_df["pred_label"]
        valid = labels.notna() & labels.isin(LABEL2ID.keys())
        if not valid.any():
            return None
        label_ids = np.full(len(self.metadata_df), -1, dtype=np.int64)
        label_ids[valid.to_numpy()] = labels.loc[valid].map(LABEL2ID).astype(np.int64).to_numpy()
        return label_ids


def build_target_dataloader(
    dataset: MALTTargetDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
