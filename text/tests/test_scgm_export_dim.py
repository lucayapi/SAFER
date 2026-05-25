"""Vérifie que les embeddings exportés SCGM ont dimension hiddim."""

import torch

from scgm_text.scgm_text_model import SCGMTextModel


def test_forward_embedding_dim_equals_hiddim():
    hiddim = 16
    model = SCGMTextModel(
        hiddim=hiddim,
        num_classes=4,
        num_subclasses=8,
        backbone_model_name_or_path="__test_dummy__",
        projection="linear",
    )
    batch = {
        "input_ids": torch.randint(1, 50, (3, 10)),
        "attention_mask": torch.ones(3, 10, dtype=torch.long),
    }
    out = model(batch)
    assert out.shape == (3, hiddim)
