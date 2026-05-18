"""Utilitaires SentenceTransformer partagés (triplet, supcon)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from transformers import TrainerCallback

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.eval_geometry import evaluate_st_val_geometry, selection_score


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_sentence_transformer(cfg: ContrastiveConfig) -> SentenceTransformer:
    model = SentenceTransformer(cfg.backbone_name, trust_remote_code=True)
    if cfg.max_seq_length:
        model.max_seq_length = int(cfg.max_seq_length)
    return model


def dataframe_to_hf_dataset(df: pd.DataFrame, text_col: str) -> Dataset:
    return Dataset.from_dict(
        {
            "sentence": df[text_col].astype(str).tolist(),
            "label": df["label_id"].astype(int).tolist(),
        }
    )


class GeometrySelectionCallback(TrainerCallback):
    """Sélection du meilleur checkpoint sur val delta_macro_pct (100×η²)."""

    def __init__(
        self,
        model: SentenceTransformer,
        val_df: pd.DataFrame,
        text_col: str,
        cfg: ContrastiveConfig,
        best_model_dir: Path,
        log_rows: List[Dict[str, Any]],
    ) -> None:
        self.model = model
        self.val_df = val_df
        self.text_col = text_col
        self.cfg = cfg
        self.best_model_dir = best_model_dir
        self.log_rows = log_rows
        self.best_score = float("-inf")
        self.best_geometry: Dict[str, Any] = {}

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch is not None else len(self.log_rows) + 1
        row = evaluate_st_val_geometry(self.model, self.val_df, self.cfg, self.text_col)
        score = selection_score(row, self.cfg.selection_metric)
        train_loss = None
        if state.log_history:
            for entry in reversed(state.log_history):
                if "loss" in entry and "eval" not in entry:
                    train_loss = entry.get("loss")
                    break
        self.log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_delta_macro_pct": row.get("delta_macro_pct"),
                "val_eta2_macro_balanced": row.get("eta2_macro_balanced"),
                "val_rankme_global": row.get("rankme_global"),
                "val_c1_global": row.get("c1_global"),
                "val_c10_global": row.get("c10_global"),
            }
        )
        if score > self.best_score:
            self.best_score = score
            self.best_geometry = dict(row)
            self.best_model_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(self.best_model_dir))
        return control


def build_training_arguments(
    cfg: ContrastiveConfig,
    output_dir: Path,
    *,
    steps_per_epoch: int,
    use_eval: bool,
) -> SentenceTransformerTrainingArguments:
    device = get_device()
    use_bf16 = (
        device.startswith("cuda")
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    use_fp16 = device.startswith("cuda") and not use_bf16
    return SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        fp16=use_fp16,
        bf16=use_bf16,
        batch_sampler=BatchSamplers.GROUP_BY_LABEL,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        logging_strategy="steps",
        logging_steps=max(1, steps_per_epoch // 2),
        report_to=[],
        seed=cfg.seed,
    )


def resolve_triplet_distance(name: str):
    if not hasattr(losses, "BatchHardTripletLossDistanceFunction"):
        raise AttributeError("BatchHardTripletLossDistanceFunction introuvable.")
    cls = losses.BatchHardTripletLossDistanceFunction
    cosine_fn = getattr(cls, "cosine_distance", None)
    euclid_fn = getattr(cls, "eucledian_distance", None) or getattr(cls, "euclidean_distance", None)
    key = (name or "cosine").strip().lower()
    mapping = {
        "cosine": cosine_fn,
        "euclidean": euclid_fn,
        "eucledian": euclid_fn,
    }
    fn = mapping.get(key)
    if fn is None:
        raise ValueError(f"Distance triplet inconnue : {name}")
    return fn


def train_st_model(
    cfg: ContrastiveConfig,
    model: SentenceTransformer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str,
    train_loss,
    checkpoints_dir: Path,
    train_log_path: Optional[Path] = None,
) -> tuple[SentenceTransformer, Dict[str, Any], float]:
    train_ds = dataframe_to_hf_dataset(train_df, text_col)
    steps_per_epoch = max(
        1,
        math.ceil(
            len(train_df)
            / max(1, cfg.batch_size * cfg.gradient_accumulation_steps)
        ),
    )
    use_eval = len(val_df) > 0 and not cfg.final_fit_full_data
    args = build_training_arguments(
        cfg, checkpoints_dir / "trainer", steps_per_epoch=steps_per_epoch, use_eval=use_eval
    )
    best_dir = checkpoints_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    log_rows: List[Dict[str, Any]] = []
    callbacks = []
    if use_eval:
        callbacks.append(
            GeometrySelectionCallback(model, val_df, text_col, cfg, best_dir, log_rows)
        )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        loss=train_loss,
        callbacks=callbacks,
    )
    trainer.train()

    if use_eval and callbacks:
        cb: GeometrySelectionCallback = callbacks[0]
        best_geometry = cb.best_geometry
        best_score = cb.best_score
        if best_geometry:
            model = SentenceTransformer(str(best_dir), trust_remote_code=True)
    else:
        model.save_pretrained(str(best_dir))
        best_geometry = {}
        best_score = float("nan")

    if train_log_path is not None:
        train_log_path.parent.mkdir(parents=True, exist_ok=True)
    if train_log_path is not None and log_rows:
        pd.DataFrame(log_rows).to_csv(train_log_path, index=False)
    elif train_log_path is not None and trainer.state.log_history:
        pd.DataFrame(trainer.state.log_history).to_csv(train_log_path, index=False)

    return model, best_geometry, best_score
