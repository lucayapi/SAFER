"""GroupKFold CV pour supervised_macro_ft."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Subset

from safer_core.kfold_eval import group_kfold_splits
from scgm_text.collate import make_text_collate_fn
from scgm_text.dataset_text_raw import TextRawDataset
from supervised_macro_ft.backbone_scaler import BackboneScaler, should_standardize_backbone
from supervised_macro_ft.checkpoint_io import save_checkpoint
from supervised_macro_ft.class_balance import (
    balanced_oversample_indices,
    resolve_train_balance,
)
from supervised_macro_ft.embedding_cache import (
    BackboneHiddenDataset,
    collate_hidden_batch,
    should_cache_backbone_embeddings,
)
from supervised_macro_ft.model import SupervisedMacroModel, model_kwargs_from_cfg
from supervised_macro_ft.run_logging import log_cv_fold_done, log_cv_fold_start
from supervised_macro_ft.train_loop import build_class_weights, evaluate_loader, fit_model

logger = logging.getLogger(__name__)


def _split_inner_group_validation(
    indices: np.ndarray,
    groups: np.ndarray,
    *,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split outer-train into inner train/validation without crossing groups."""
    if not 0.0 < float(val_ratio) < 1.0:
        raise ValueError("early_stopping_inner_val_ratio doit être entre 0 et 1")
    outer_groups = np.asarray(groups)[indices].astype(str)
    if len(np.unique(outer_groups)) < 2:
        raise ValueError("La validation interne nécessite au moins deux groupes.")
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=float(val_ratio),
        random_state=int(seed),
    )
    relative_train, relative_val = next(
        splitter.split(np.zeros(len(indices)), groups=outer_groups)
    )
    return indices[relative_train].astype(np.int64), indices[relative_val].astype(np.int64)


def _count_groups(groups: np.ndarray, indices: np.ndarray) -> int:
    return int(len(np.unique(np.asarray(groups)[indices].astype(str))))


def aggregate_cv_metrics(fold_rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(list(fold_rows))
    if df.empty:
        return df
    metrics = [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "loss",
        "n_train",
        "n_train_raw",
        "n_val",
        "n_outer_train",
        "n_outer_val",
        "n_inner_train",
        "n_inner_val",
        "n_groups_outer_train",
        "n_groups_outer_val",
        "n_groups_inner_train",
        "n_groups_inner_val",
        "best_epoch",
    ]
    rows: List[Dict[str, Any]] = []
    for model_key in sorted(df["model"].unique()) if "model" in df.columns else ["supervised_macro_ft"]:
        sub = df[df["model"] == model_key] if "model" in df.columns else df
        row: Dict[str, Any] = {"model": model_key, "n_folds": len(sub)}
        for m in metrics:
            if m in sub.columns:
                row[f"mean_{m}"] = float(sub[m].mean())
                row[f"std_{m}"] = float(sub[m].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def run_group_kfold_cv(
    dataset: TextRawDataset,
    tokenizer,
    *,
    model_cfg: Mapping[str, Any],
    train_cfg: Mapping[str, Any],
    n_folds: int,
    seed: int,
    device: torch.device,
    fold_out_root: Optional[str] = None,
    backbone_hidden: Optional[np.ndarray] = None,
    save_fold_checkpoints: bool = False,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Outer GroupKFold with an inner grouped validation for early stopping."""
    groups = dataset.get_groups()
    splits = group_kfold_splits(groups, n_folds, seed)
    fold_rows: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    batch_size = int(train_cfg.get("batch_size", 32))
    max_length = int(model_cfg.get("max_seq_length", 256))
    use_hidden_cache = backbone_hidden is not None and should_cache_backbone_embeddings(model_cfg)
    collate_fn = make_text_collate_fn(tokenizer, max_length)
    label_ids = dataset.label_ids
    use_oversampling, class_weight_mode = resolve_train_balance(model_cfg)
    selection_metric = str(train_cfg.get("selection_metric", "balanced_accuracy"))
    inner_val_ratio = float(train_cfg.get("early_stopping_inner_val_ratio", 0.1))

    for fold_id, (train_idx, val_idx) in enumerate(splits):
        inner_train_idx, inner_val_idx = _split_inner_group_validation(
            train_idx,
            groups,
            val_ratio=inner_val_ratio,
            seed=seed + int(fold_id),
        )
        train_idx_fit = inner_train_idx
        if use_oversampling:
            train_idx_fit = balanced_oversample_indices(
                label_ids, inner_train_idx, seed=seed + int(fold_id)
            )
        log_cv_fold_start(
            fold_id,
            n_folds,
            n_train=int(len(train_idx_fit)),
            n_val=int(len(val_idx)),
            oversampled=bool(use_oversampling and len(train_idx_fit) != len(train_idx)),
            use_hidden_cache=use_hidden_cache,
        )
        model = SupervisedMacroModel(**model_kwargs_from_cfg(model_cfg)).to(device)

        if use_hidden_cache and should_standardize_backbone(model_cfg):
            assert backbone_hidden is not None
            scaler = BackboneScaler.fit(backbone_hidden, inner_train_idx)
            model.set_backbone_scaler(scaler)

        if use_hidden_cache:
            assert backbone_hidden is not None
            train_ds = BackboneHiddenDataset(backbone_hidden, label_ids, train_idx_fit)
            inner_val_ds = BackboneHiddenDataset(backbone_hidden, label_ids, inner_val_idx)
            outer_val_ds = BackboneHiddenDataset(backbone_hidden, label_ids, val_idx)
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_hidden_batch
            )
            inner_val_loader = DataLoader(
                inner_val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_hidden_batch
            )
            outer_val_loader = DataLoader(
                outer_val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_hidden_batch
            )
        else:
            train_ds = Subset(dataset, train_idx_fit.tolist())
            inner_val_ds = Subset(dataset, inner_val_idx.tolist())
            outer_val_ds = Subset(dataset, val_idx.tolist())
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            inner_val_loader = DataLoader(
                inner_val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
            )
            outer_val_loader = DataLoader(
                outer_val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
            )

        train_labels = [int(dataset.label_ids[i]) for i in inner_train_idx]
        class_weight = build_class_weights(
            train_labels if not use_oversampling else [int(dataset.label_ids[i]) for i in train_idx_fit],
            int(model_cfg.get("n_classes", 4)),
            class_weight_mode,
        )

        model, metrics, fold_history = fit_model(
            model,
            train_loader,
            inner_val_loader,
            train_cfg=dict(train_cfg),
            device=device,
            class_weight=class_weight,
            run_label=f"cv_fold_{fold_id}",
        )
        outer_metrics = evaluate_loader(model, outer_val_loader, device)
        for hist_row in fold_history:
            history_rows.append(
                {
                    "phase": "cv",
                    "fold": fold_id,
                    "validation_split": "inner",
                    **hist_row,
                }
            )
        row = {
            "fold": fold_id,
            "model": "supervised_macro_ft",
            **outer_metrics,
            "inner_val_loss": metrics.get("val_loss"),
            "inner_val_accuracy": metrics.get("val_accuracy"),
            "inner_val_macro_f1": metrics.get("val_macro_f1"),
            "inner_val_balanced_accuracy": metrics.get("val_balanced_accuracy"),
            "best_epoch": metrics.get("epoch"),
            "max_epochs": int(train_cfg.get("epochs", 30)),
            "early_stopped": bool(
                metrics.get("epoch") is not None
                and int(metrics["epoch"]) < int(train_cfg.get("epochs", 30))
            ),
            "n_train": int(len(train_idx_fit)),
            "n_train_raw": int(len(inner_train_idx)),
            "n_val": int(len(val_idx)),
            "n_outer_train": int(len(train_idx)),
            "n_outer_val": int(len(val_idx)),
            "n_inner_train": int(len(inner_train_idx)),
            "n_inner_val": int(len(inner_val_idx)),
            "n_groups_outer_train": _count_groups(groups, train_idx),
            "n_groups_outer_val": _count_groups(groups, val_idx),
            "n_groups_inner_train": _count_groups(groups, inner_train_idx),
            "n_groups_inner_val": _count_groups(groups, inner_val_idx),
        }
        fold_rows.append(row)
        log_cv_fold_done(fold_id, n_folds, metrics, selection_metric=selection_metric)

        if fold_out_root and save_fold_checkpoints:
            fold_dir = f"{fold_out_root}/folds/fold_{fold_id}"
            save_checkpoint(
                model,
                fold_dir,
                config={**dict(model_cfg), **dict(train_cfg), "fold": fold_id},
            )

    history_df = pd.DataFrame(history_rows)
    return fold_rows, aggregate_cv_metrics(fold_rows), history_df
