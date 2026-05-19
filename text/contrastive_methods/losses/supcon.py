"""SupCon via HobbitLong/SupContrast (réexport pour compatibilité imports)."""

from contrastive_methods.losses.supcon_hobbit import (
    HobbitSupConLoss,
    SupConSentenceTransformerLoss,
    build_supcon_loss,
)

# Alias historique : la loss ST utilisée à l'entraînement
SupConLoss = SupConSentenceTransformerLoss

__all__ = [
    "HobbitSupConLoss",
    "SupConLoss",
    "SupConSentenceTransformerLoss",
    "build_supcon_loss",
]
