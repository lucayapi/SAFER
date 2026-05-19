"""Ligne de tableau géométrique (eta² + RankMe + C1/C10)."""

from __future__ import annotations

from metrics.embedding_geometry_separation import (
    GEOMETRY_METRIC_KEYS,
    METRICS_TABLE_COLUMNS,
    PRIMARY_SELECTION_METRIC,
    build_geometry_metrics_row,
    metrics_table_from_rows,
)

__all__ = [
    "GEOMETRY_METRIC_KEYS",
    "METRICS_TABLE_COLUMNS",
    "PRIMARY_SELECTION_METRIC",
    "build_geometry_metrics_row",
    "metrics_table_from_rows",
]
