"""SCGM-G adaptation for fixed text embeddings."""

from scgm_text.dataset_text_embeddings import TextEmbeddingDataset, split_by_group
from scgm_text.fidelity import (
    apply_scgm_strict_defaults,
    apply_text_pragmatic_defaults,
    describe_fidelity_mode,
)
from scgm_text.scgm_embedding_model import SCGMEmbeddingNet

__all__ = [
    "TextEmbeddingDataset",
    "split_by_group",
    "SCGMEmbeddingNet",
    "apply_scgm_strict_defaults",
    "apply_text_pragmatic_defaults",
    "describe_fidelity_mode",
]
