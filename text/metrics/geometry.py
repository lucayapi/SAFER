"""Ligne de tableau géométrique (η² macro, δ, IPR, etc.)."""

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
