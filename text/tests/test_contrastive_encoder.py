"""Tests encodeur contrastif unifié."""

from __future__ import annotations

import sys
from pathlib import Path

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.encoder_model import ContrastiveEncoder, EncoderConfig


def test_contrastive_encoder_frozen_with_projector():
    enc = ContrastiveEncoder(
        EncoderConfig(
            backbone_name="__test_dummy__",
            use_projector=True,
            projection="linear",
            hiddim=16,
            backbone_trainable=False,
        )
    )
    assert enc.embedding_dim == 16
    assert enc.cache_backbone_embeddings is True
    assert enc.projector is not None


def test_contrastive_encoder_no_projector():
    enc = ContrastiveEncoder(
        EncoderConfig(
            backbone_name="__test_dummy__",
            use_projector=False,
            backbone_trainable=False,
        )
    )
    assert enc.embedding_dim == 32
    assert enc.projector is None


def test_contrastive_encoder_forward_dummy():
    import torch

    enc = ContrastiveEncoder(
        EncoderConfig(
            backbone_name="__test_dummy__",
            use_projector=True,
            projection="linear",
            hiddim=8,
        )
    )
    out = enc(
        {
            "input_ids": torch.randint(0, 10, (4, 6)),
            "attention_mask": torch.ones(4, 6, dtype=torch.long),
        }
    )
    assert out.shape == (4, 8)
