"""Orchestration entraînement supervised_macro_ft (CV + fit final)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from macro_transfer.encode import load_target_metadata
from safer_core.io import load_yaml
from safer_core.paths import resolve_repo_path
from safer_core.test_corpus import resolve_test_corpus
from scgm_text.collate import make_text_collate_fn
from scgm_text.dataset_text_raw import TextRawDataset
from supervised_macro_ft.checkpoint_io import save_checkpoint
from supervised_macro_ft.cv import run_group_kfold_cv
from supervised_macro_ft.model import SupervisedMacroModel, model_kwargs_from_cfg
from supervised_macro_ft.train_loop import (
    build_class_weights,
    evaluate_loader,
    fit_model,
)

logger = logging.getLogger(__name__)


def _load_tokenizer(backbone_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(backbone_name, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    return tok


def run_supervised_macro_ft_training(config_path: str | Path) -> Dict[str, Any]:
    anchor = Path(__file__).resolve().parents[1]
    cfg = load_yaml(Path(config_path))
    data_cfg = dict(cfg.get("data") or {})
    model_cfg = dict(cfg.get("model") or {})
    train_cfg = dict(cfg.get("training") or {})
    out_dir = resolve_repo_path(str(cfg.get("output_dir", "output/supervised_macro_ft")), repo_root=anchor)
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone_name = str(model_cfg.get("backbone_name", "Qwen/Qwen3-Embedding-0.6B"))
    tokenizer = _load_tokenizer(backbone_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_csv = resolve_repo_path(data_cfg.get("data_csv", "dataset/data_btp.csv"), repo_root=anchor)
    dataset = TextRawDataset(
        str(data_csv),
        label_col=str(data_cfg.get("label_col", "pred_label")),
        pred_ok_col=str(data_cfg.get("pred_ok_col", "pred_ok")),
        group_col=str(data_cfg.get("group_col", "accident_id")),
        text_col=data_cfg.get("text_col"),
    )

    n_folds = int(train_cfg.get("n_folds", 5))
    seed = int(train_cfg.get("seed", 42))
    fold_rows, cv_summary, cv_history = run_group_kfold_cv(
        dataset,
        tokenizer,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        n_folds=n_folds,
        seed=seed,
        device=device,
        fold_out_root=str(out_dir),
    )
    cv_dir = out_dir / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(cv_dir / "cv_per_fold.csv", index=False)
    cv_summary.to_csv(cv_dir / "cv_summary.csv", index=False)
    cv_summary.to_csv(out_dir / "kfold_summary.csv", index=False)
    if not cv_history.empty:
        cv_history.to_csv(cv_dir / "train_history.csv", index=False)
        logger.info("Historique CV exporté : %s", cv_dir / "train_history.csv")

    # Fit final 100 % BTP
    batch_size = int(train_cfg.get("batch_size", 32))
    max_length = int(model_cfg.get("max_seq_length", 256))
    collate_fn = make_text_collate_fn(tokenizer, max_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = SupervisedMacroModel(**model_kwargs_from_cfg(model_cfg)).to(device)

    class_weight = build_class_weights(
        dataset.label_ids.tolist(),
        int(model_cfg.get("n_classes", 4)),
        model_cfg.get("class_weight"),
    )
    model, final_metrics, final_history = fit_model(
        model,
        train_loader,
        val_loader=None,
        train_cfg=train_cfg,
        device=device,
        class_weight=class_weight,
        run_label="final_fit",
    )
    if final_history:
        final_hist_df = pd.DataFrame(
            [{"phase": "final", "fold": -1, **row} for row in final_history]
        )
        final_hist_df.to_csv(out_dir / "train_history_final.csv", index=False)
        logger.info("Historique fit final exporté : %s", out_dir / "train_history_final.csv")

    ckpt_dir = out_dir / "checkpoints" / "best_model"
    save_checkpoint(
        model,
        ckpt_dir,
        config={**model_cfg, **train_cfg, "method_name": cfg.get("method_name", "supervised_macro_ft")},
    )

    exp_cfg = dict(cfg.get("exports") or {})
    if bool(exp_cfg.get("save_btp_embeddings", True)):
        from macro_transfer.constants import MACRO_NAMES
        from supervised_macro_ft.transfer import encode_texts, predict_corpus

        emb_dir = out_dir / "embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)
        meta_df = dataset.get_metadata_df()
        text_col = str(data_cfg.get("text_col", "sentence"))
        label_col = str(data_cfg.get("label_col", "pred_label"))
        group_col = str(data_cfg.get("group_col", "accident_id"))
        texts = meta_df[text_col].astype(str).tolist()
        h_btp = encode_texts(
            model,
            tokenizer,
            texts,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )
        pred_macro, probs, confidence, margin, entropy = predict_corpus(
            model,
            tokenizer,
            texts,
            macros=list(MACRO_NAMES),
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )
        btp_preds = pd.DataFrame(
            {
                "sentence": texts,
                label_col: meta_df[label_col].astype(str).to_numpy(),
                "pred_macro": pred_macro,
                "confidence": confidence,
                "margin": margin,
                "entropy": entropy,
            }
        )
        if group_col in meta_df.columns:
            btp_preds[group_col] = meta_df[group_col].to_numpy()
        for i, m in enumerate(MACRO_NAMES):
            btp_preds[f"prob_{m}"] = probs[:, i]
        np.save(emb_dir / "btp_embeddings.npy", h_btp)
        btp_preds.to_csv(emb_dir / "btp_embeddings_metadata.csv", index=False)
        logger.info("Embeddings BTP exportés : %s", emb_dir)

    # Eval optionnelle corpus test
    test_metrics: Dict[str, Any] = {}
    test_corpus = str(cfg.get("test_corpus") or "metallurgie")
    try:
        from macro_transfer.constants import LABEL2ID

        spec = resolve_test_corpus(test_corpus, anchor=anchor)
        test_meta = load_target_metadata(str(spec.data_csv), text_col=str(data_cfg.get("text_col", "sentence")))
        label_col = str(data_cfg.get("label_col", "pred_label"))
        test_meta = test_meta.copy()
        if label_col in test_meta.columns:
            test_meta["label_id"] = test_meta[label_col].astype(str).map(LABEL2ID)
        else:
            test_meta["label_id"] = 0
        test_ds = TextRawDataset(
            str(spec.data_csv),
            label_col=label_col,
            group_col=str(data_cfg.get("group_col", "accident_id")),
            text_col=data_cfg.get("text_col"),
            metadata_df=test_meta,
        )
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_metrics = evaluate_loader(model, test_loader, device)
        pd.DataFrame([test_metrics]).to_csv(out_dir / "metrics_geometry_test.csv", index=False)
    except Exception as exc:
        logger.warning("Eval test corpus ignorée : %s", exc)

    result = {
        "output_dir": str(out_dir),
        "checkpoint_dir": str(ckpt_dir),
        "cv_summary": cv_summary.to_dict(orient="records"),
        "final_fit_metrics": final_metrics,
        "test_metrics": test_metrics,
        "train_history_paths": {
            "cv": str(cv_dir / "train_history.csv") if not cv_history.empty else None,
            "final": str(out_dir / "train_history_final.csv") if final_history else None,
        },
    }
    with open(out_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result
