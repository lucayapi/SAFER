"""Chargement CSV et split train/val par groupe."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from contrastive_methods.config import ContrastiveConfig
from scgm_text.dataset_text_raw import TextRawDataset, split_raw_by_group


def prepare_text_dataset(cfg: ContrastiveConfig) -> TextRawDataset:
    return TextRawDataset(
        data_csv=str(cfg.dataset_path),
        label_col=cfg.label_col,
        pred_ok_col=cfg.pred_ok_col,
        group_col=cfg.group_col,
        text_col=cfg.text_col,
    )


def split_train_val(
    dataset: TextRawDataset,
    cfg: ContrastiveConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    fold_train = cfg.extra.get("fold_train_idx")
    fold_val = cfg.extra.get("fold_val_idx")
    if fold_train is not None and fold_val is not None:
        return np.asarray(fold_train, dtype=np.int64), np.asarray(fold_val, dtype=np.int64)
    if cfg.final_fit_full_data:
        n = len(dataset)
        all_idx = np.arange(n, dtype=np.int64)
        return all_idx, np.array([], dtype=np.int64)
    return split_raw_by_group(dataset, val_ratio=cfg.val_ratio, seed=cfg.seed)


def get_group_kfold_splits(dataset: TextRawDataset, cfg: ContrastiveConfig):
    from safer_core.kfold_eval import group_kfold_splits

    groups = dataset.get_groups()
    n_folds = int(cfg.n_folds) if cfg.n_folds and cfg.n_folds > 1 else 5
    return group_kfold_splits(groups, n_folds, cfg.seed)


def train_val_metadata(
    dataset: TextRawDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    meta = dataset.get_metadata_df()
    return meta.iloc[train_idx].copy(), meta.iloc[val_idx].copy()
