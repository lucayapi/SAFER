"""SupCon via HobbitLong/SupContrast (réexport)."""

from contrastive_methods.losses.supcon_hobbit import (
    HobbitSupConLoss,
    SupConEmbeddingLoss,
    build_supcon_embedding_loss,
)

SupConLoss = SupConEmbeddingLoss
build_supcon_loss = build_supcon_embedding_loss

__all__ = [
    "HobbitSupConLoss",
    "SupConLoss",
    "SupConEmbeddingLoss",
    "build_supcon_embedding_loss",
    "build_supcon_loss",
]
