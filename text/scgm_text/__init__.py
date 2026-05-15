"""SCGM-G adaptation for fixed text embeddings."""

from scgm_text.dataset_text_embeddings import TextEmbeddingDataset, split_by_group
from scgm_text.scgm_embedding_model import SCGMEmbeddingNet

__all__ = ["TextEmbeddingDataset", "split_by_group", "SCGMEmbeddingNet"]
