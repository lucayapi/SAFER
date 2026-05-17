"""Métriques d'inertie eta² (macro-balanced et weighted)."""

from __future__ import annotations

from metrics.embedding_geometry_separation import (
    MACRO_NAMES,
    compute_eta2_macro_geometry,
)

__all__ = ["MACRO_NAMES", "compute_eta2_inertia_metrics", "compute_eta2_macro_geometry"]


def compute_eta2_inertia_metrics(
    embeddings,
    labels,
    labels_order=MACRO_NAMES,
    l2_normalize: bool = False,
    eps: float = 1e-12,
):
    """Alias documenté du plan — délègue à ``compute_eta2_macro_geometry``."""
    return compute_eta2_macro_geometry(
        embeddings,
        labels,
        labels=labels_order,
        l2_normalize=l2_normalize,
        eps=eps,
    )
