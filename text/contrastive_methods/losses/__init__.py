"""Losses contrastives natives."""

from contrastive_methods.losses.softtriple import SoftTripleLoss
from contrastive_methods.losses.supcon import SupConLoss, build_supcon_embedding_loss
from contrastive_methods.losses.supcon_hobbit import HobbitSupConLoss, SupConEmbeddingLoss
from contrastive_methods.losses.triplet_st import (
    BatchTripletEmbeddingLoss,
    build_batch_triplet_embedding_loss,
)

__all__ = [
    "SoftTripleLoss",
    "HobbitSupConLoss",
    "SupConLoss",
    "SupConEmbeddingLoss",
    "build_supcon_embedding_loss",
    "BatchTripletEmbeddingLoss",
    "build_batch_triplet_embedding_loss",
]
