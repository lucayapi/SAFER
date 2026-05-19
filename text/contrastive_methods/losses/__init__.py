"""Losses contrastives natives."""

from contrastive_methods.losses.softtriple import HFTextEncoder, SoftTripleLoss
from contrastive_methods.losses.supcon import SupConLoss
from contrastive_methods.losses.supcon_hobbit import HobbitSupConLoss, build_supcon_loss

__all__ = [
    "HFTextEncoder",
    "SoftTripleLoss",
    "HobbitSupConLoss",
    "SupConLoss",
    "build_supcon_loss",
    "build_batch_hard_soft_margin_loss",
]


def __getattr__(name: str):
    if name == "build_batch_hard_soft_margin_loss":
        from contrastive_methods.losses.triplet_st import build_batch_hard_soft_margin_loss

        return build_batch_hard_soft_margin_loss
    raise AttributeError(name)
