"""Utilitaires SentenceTransformer partagés (triplet, supcon)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer

from contrastive_methods.config import ContrastiveConfig


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


def build_training_arguments(
    cfg: ContrastiveConfig,
    output_dir: Path,
    *,
    steps_per_epoch: int,
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
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
) -> SentenceTransformer:
    train_ds = dataframe_to_hf_dataset(train_df, text_col)
    val_ds = dataframe_to_hf_dataset(val_df, text_col)
    steps_per_epoch = max(
        1,
        math.ceil(
            len(train_df)
            / max(1, cfg.batch_size * cfg.gradient_accumulation_steps)
        ),
    )
    args = build_training_arguments(cfg, checkpoints_dir / "trainer", steps_per_epoch=steps_per_epoch)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        loss=train_loss,
    )
    trainer.train()
    if train_log_path is not None:
        log_hist = trainer.state.log_history
        if log_hist:
            pd.DataFrame(log_hist).to_csv(train_log_path, index=False)
    best_dir = checkpoints_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(best_dir))
    return trainer.model
