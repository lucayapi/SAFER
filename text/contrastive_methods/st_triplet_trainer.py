"""Trainer SentenceTransformer contrastif (PK triplet + suivi loss)."""

from __future__ import annotations

from typing import Any, Optional

import torch
from datasets import Dataset
from sentence_transformers.trainer import SentenceTransformerTrainer
from torch.utils.data import BatchSampler

from contrastive_methods.samplers.pk_batch_sampler import PKBatchSampler, PKParams
from contrastive_methods.training_log import EpochLossAccumulator


class ContrastiveSTTrainer(SentenceTransformerTrainer):
    """
    Trainer ST pour SupCon (shuffle) et Batch Triplet (PK optionnel).

    Accumule la loss par epoch sans logs HF step (``logging_strategy=no``).
    """

    def __init__(
        self,
        *args: Any,
        pk_params: Optional[PKParams] = None,
        loss_accumulator: Optional[EpochLossAccumulator] = None,
        **kwargs: Any,
    ) -> None:
        self.pk_params = pk_params
        self.loss_accumulator = loss_accumulator or EpochLossAccumulator()
        super().__init__(*args, **kwargs)

    def get_batch_sampler(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: list[str] | None = None,
        generator: torch.Generator | None = None,
    ) -> BatchSampler | None:
        if self.pk_params is not None and (self.pk_params.sampler or "").strip().lower() == "pk":
            label_col = "label"
            if valid_label_columns:
                for col in valid_label_columns:
                    if col in dataset.column_names:
                        label_col = col
                        break
            if label_col not in dataset.column_names:
                raise ValueError(
                    f"Colonne label {label_col!r} absente du dataset "
                    f"(colonnes : {dataset.column_names})."
                )
            return PKBatchSampler(
                labels=dataset[label_col],
                batch_size=self.pk_params.batch_size,
                classes_per_batch=self.pk_params.classes_per_batch,
                samples_per_class=self.pk_params.samples_per_class,
                drop_last=self.pk_params.drop_last,
                seed=self.pk_params.seed,
            )
        return super().get_batch_sampler(
            dataset,
            batch_size,
            drop_last,
            valid_label_columns=valid_label_columns,
            generator=generator,
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        loss = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        if return_outputs:
            loss_val, outputs = loss
            self.loss_accumulator.record(float(loss_val.detach().item()), self.state.epoch)
            return loss_val, outputs
        self.loss_accumulator.record(float(loss.detach().item()), self.state.epoch)
        return loss
