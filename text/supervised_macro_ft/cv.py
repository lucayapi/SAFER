"""GroupKFold CV pour supervised_macro_ft."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from safer_core.kfold_eval import group_kfold_splits
from scgm_text.collate import make_text_collate_fn
from scgm_text.dataset_text_raw import TextRawDataset
from supervised_macro_ft.checkpoint_io import save_checkpoint
from supervised_macro_ft.model import SupervisedMacroModel
from supervised_macro_ft.train_loop import build_class_weights, evaluate_loader, fit_model


def aggregate_cv_metrics(fold_rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(list(fold_rows))
    if df.empty:
        return df
    metrics = ["accuracy", "macro_f1", "balanced_accuracy", "loss"]
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
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    groups = dataset.get_groups()
    splits = group_kfold_splits(groups, n_folds, seed)
    fold_rows: List[Dict[str, Any]] = []
    batch_size = int(train_cfg.get("batch_size", 32))
    max_length = int(model_cfg.get("max_seq_length", 256))
    collate_fn = make_text_collate_fn(tokenizer, max_length)

    for fold_id, (train_idx, val_idx) in enumerate(splits):
        train_ds = Subset(dataset, train_idx.tolist())
        val_ds = Subset(dataset, val_idx.tolist())
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        model = SupervisedMacroModel(
            backbone_name=str(model_cfg["backbone_name"]),
            num_classes=int(model_cfg.get("n_classes", 4)),
            pooling=str(model_cfg.get("pooling", "mean")),
            backbone_trainable=bool(model_cfg.get("backbone_trainable", True)),
            train_last_n_layers=model_cfg.get("train_last_n_layers"),
            gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", False)),
        ).to(device)

        train_labels = [int(dataset.label_ids[i]) for i in train_idx]
        class_weight = build_class_weights(
            train_labels,
            int(model_cfg.get("n_classes", 4)),
            model_cfg.get("class_weight"),
        )

        model, metrics = fit_model(
            model,
            train_loader,
            val_loader,
            train_cfg=dict(train_cfg),
            device=device,
            class_weight=class_weight,
        )
        row = {
            "fold": fold_id,
            "model": "supervised_macro_ft",
            **metrics,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
        }
        fold_rows.append(row)

        if fold_out_root:
            fold_dir = f"{fold_out_root}/folds/fold_{fold_id}"
            save_checkpoint(
                model,
                fold_dir,
                config={**dict(model_cfg), **dict(train_cfg), "fold": fold_id},
            )

    return fold_rows, aggregate_cv_metrics(fold_rows)
