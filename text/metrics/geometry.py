"""Ligne de tableau géométrique (eta² + RankMe + C1/C10)."""

from __future__ import annotations

from metrics.embedding_geometry_separation import (
    METRICS_TABLE_COLUMNS,
    build_geometry_metrics_row,
    metrics_table_from_rows,
)

__all__ = [
    "METRICS_TABLE_COLUMNS",
    "build_geometry_metrics_row",
    "metrics_table_from_rows",
]
