"""Collate batches pour pipeline SCGM end2end (texte tokenisé)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List


import torch


def make_text_collate_fn(tokenizer, max_length: int) -> Callable:
    def collate_text_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [item["text"] for item in batch]
        label_ids = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
        indices = torch.tensor([int(item["index"]) for item in batch], dtype=torch.long)
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
            "labels": label_ids,
            "indices": indices,
        }

    return collate_text_batch
