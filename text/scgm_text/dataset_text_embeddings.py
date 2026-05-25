"""Shim: frozen-embedding dataset moved to scgm_text.legacy."""

from scgm_text.legacy.dataset_text_embeddings import (  # noqa: F401
    TextEmbeddingDataset,
    build_dataloaders,
    merge_metadata_with_embeddings,
    split_by_group,
)
from scgm_text.data_metadata import LABEL2ID, ID2LABEL, VALID_LABELS, load_filtered_metadata

__all__ = [
    "LABEL2ID",
    "ID2LABEL",
    "VALID_LABELS",
    "TextEmbeddingDataset",
    "build_dataloaders",
    "load_filtered_metadata",
    "merge_metadata_with_embeddings",
    "split_by_group",
]
