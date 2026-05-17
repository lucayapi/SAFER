"""Alias rétrocompatible : embeddings pré-calculés + tête SCGM."""

from __future__ import annotations

from typing import Optional

from scgm_text.scgm_head import glorot  # noqa: F401 — utilisé par malt_model
from scgm_text.scgm_text_model import SCGMTextModel

__all__ = ["SCGMEmbeddingNet", "glorot"]


class SCGMEmbeddingNet(SCGMTextModel):
    """
    SCGM-G sur embeddings pré-calculés (``input_mode=precomputed_embeddings``).

  ``projection=identity`` : pas de couche linéaire additionnelle sur e ; les vecteurs e
    restent ceux du CSV (backbone hors graphe). Pour fine-tuner f_theta(x), utiliser
    ``SCGMTextModel`` avec ``input_mode=text``.
    """

    def __init__(
        self,
        input_dim: int,
        hiddim: int,
        num_classes: int,
        num_subclasses: int,
        projection: str = "identity",
        dropout: float = 0.0,
        with_mlp: Optional[bool] = None,
        kd_t: float = 4.0,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hiddim=hiddim,
            num_classes=num_classes,
            num_subclasses=num_subclasses,
            projection=projection,
            dropout=dropout,
            with_mlp=with_mlp,
            kd_t=kd_t,
            input_mode="precomputed_embeddings",
        )

    def forward(self, x):  # type: ignore[override]
        return super().forward(x)
