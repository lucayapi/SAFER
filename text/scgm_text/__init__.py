"""SCGM-G adaptation for text (backbone fine-tune) or precomputed embeddings."""

from scgm_text.dataset_text_embeddings import TextEmbeddingDataset, split_by_group
from scgm_text.fidelity import (
    apply_scgm_strict_defaults,
    apply_strict_finetune_identity_defaults,
    apply_text_pragmatic_defaults,
    describe_fidelity_mode,
)
from scgm_text.scgm_embedding_model import SCGMEmbeddingNet
from scgm_text.scgm_text_model import SCGMTextModel

__all__ = [
    "TextEmbeddingDataset",
    "split_by_group",
    "SCGMEmbeddingNet",
    "SCGMTextModel",
    "apply_scgm_strict_defaults",
    "apply_strict_finetune_identity_defaults",
    "apply_text_pragmatic_defaults",
    "describe_fidelity_mode",
]
