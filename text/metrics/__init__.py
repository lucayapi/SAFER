"""Métriques SAFER (géométrie, inertie, topics)."""

from metrics.geometry import (
    GEOMETRY_METRIC_KEYS,
    METRICS_TABLE_COLUMNS,
    PRIMARY_SELECTION_METRIC,
    build_geometry_metrics_row,
    metrics_table_from_rows,
)
from metrics.inertia import compute_eta2_inertia_metrics, compute_eta2_macro_geometry
from metrics.rankme import pca_energy_c1_c10, rankme_effective_rank

__all__ = [
    "GEOMETRY_METRIC_KEYS",
    "METRICS_TABLE_COLUMNS",
    "PRIMARY_SELECTION_METRIC",
    "build_geometry_metrics_row",
    "metrics_table_from_rows",
    "compute_eta2_inertia_metrics",
    "compute_eta2_macro_geometry",
    "rankme_effective_rank",
    "pca_energy_c1_c10",
]
