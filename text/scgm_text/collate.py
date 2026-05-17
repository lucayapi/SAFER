"""Collate batches pour mode texte."""

from __future__ import annotations

from typing import Any, Callable, List, Tuple

import torch


def make_text_collate_fn(tokenizer, max_length: int) -> Callable:
    def collate_text_batch(
        batch: List[Tuple[str, torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        texts = [item[0] for item in batch]
        label_ids = torch.stack([item[1] for item in batch])
        indices = torch.stack([item[2] for item in batch])
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "label_ids": label_ids,
            "indices": indices,
        }

    return collate_text_batch
