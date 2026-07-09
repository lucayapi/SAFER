"""GroupKFold CV pour supervised_macro_ft."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
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
from supervised_macro_ft.geometry_eval import encode_val_projected, evaluate_fold_geometry
from supervised_macro_ft.model import SupervisedMacroModel, model_kwargs_from_cfg
from supervised_macro_ft.run_logging import log_cv_fold_done, log_cv_fold_start
from supervised_macro_ft.train_loop import build_class_weights, evaluate_loader, fit_model

logger = logging.getLogger(__name__)


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
    backbone_hidden: Optional[np.ndarray] = None,
    label_col: str = "pred_label",
    raw_emb_csv: Optional[str] = None,
    save_fold_checkpoints: bool = True,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    groups = dataset.get_groups()
    splits = group_kfold_splits(groups, n_folds, seed)
    fold_rows: List[Dict[str, Any]] = []
    geometry_fold_rows: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    batch_size = int(train_cfg.get("batch_size", 32))
    max_length = int(model_cfg.get("max_seq_length", 256))
    use_hidden_cache = backbone_hidden is not None and should_cache_backbone_embeddings(model_cfg)
    collate_fn = make_text_collate_fn(tokenizer, max_length)
    label_ids = dataset.label_ids
    use_oversampling, class_weight_mode = resolve_train_balance(model_cfg)
    selection_metric = str(train_cfg.get("selection_metric", "balanced_accuracy"))

    for fold_id, (train_idx, val_idx) in enumerate(splits):
        train_idx_fit = train_idx
        if use_oversampling:
            train_idx_fit = balanced_oversample_indices(
                label_ids, train_idx, seed=seed + int(fold_id)
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
            scaler = BackboneScaler.fit(backbone_hidden, train_idx)
            model.set_backbone_scaler(scaler)

        if use_hidden_cache:
            assert backbone_hidden is not None
            train_ds = BackboneHiddenDataset(backbone_hidden, label_ids, train_idx_fit)
            val_ds = BackboneHiddenDataset(backbone_hidden, label_ids, val_idx)
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_hidden_batch
            )
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_hidden_batch
            )
        else:
            train_ds = Subset(dataset, train_idx_fit.tolist())
            val_ds = Subset(dataset, val_idx.tolist())
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        train_labels = [int(dataset.label_ids[i]) for i in train_idx]
        class_weight = build_class_weights(
            train_labels if not use_oversampling else [int(dataset.label_ids[i]) for i in train_idx_fit],
            int(model_cfg.get("n_classes", 4)),
            class_weight_mode,
        )

        model, metrics, fold_history = fit_model(
            model,
            train_loader,
            val_loader,
            train_cfg=dict(train_cfg),
            device=device,
            class_weight=class_weight,
            run_label=f"cv_fold_{fold_id}",
        )
        for hist_row in fold_history:
            history_rows.append(
                {
                    "phase": "cv",
                    "fold": fold_id,
                    **hist_row,
                }
            )
        row = {
            "fold": fold_id,
            "model": "supervised_macro_ft",
            **metrics,
            "n_train": int(len(train_idx_fit)),
            "n_train_raw": int(len(train_idx)),
            "n_val": int(len(val_idx)),
        }
        fold_rows.append(row)
        log_cv_fold_done(fold_id, n_folds, metrics, selection_metric=selection_metric)

        val_meta = dataset.get_metadata_df().iloc[val_idx]
        z_val = encode_val_projected(
            model,
            dataset,
            val_idx,
            backbone_hidden=backbone_hidden if use_hidden_cache else None,
            tokenizer=tokenizer,
            device=device,
            model_cfg=model_cfg,
            batch_size=batch_size,
        )
        geom = evaluate_fold_geometry(
            z_val,
            val_meta,
            label_col,
            raw_emb_csv=raw_emb_csv,
        )
        geometry_fold_rows.append({"fold": fold_id, "fold_id": fold_id, **geom})

        if fold_out_root and save_fold_checkpoints:
            fold_dir = f"{fold_out_root}/folds/fold_{fold_id}"
            save_checkpoint(
                model,
                fold_dir,
                config={**dict(model_cfg), **dict(train_cfg), "fold": fold_id},
            )

    history_df = pd.DataFrame(history_rows)
    return fold_rows, aggregate_cv_metrics(fold_rows), history_df, geometry_fold_rows
