"""Legacy support--lift Pareto helpers (diagnostic / reproducibility only).

NOT used by the primary recurrent-scenario mining pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def is_pareto_efficient(points: np.ndarray) -> np.ndarray:
    """Return a boolean mask of non-dominated points for maximize-all objectives."""
    n = len(points)
    if n == 0:
        return np.array([], dtype=bool)
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        dominates = np.all(points >= points[i], axis=1) & np.any(points > points[i], axis=1)
        dominates[i] = False
        if dominates.any():
            is_efficient[i] = False
    return is_efficient


def mark_support_lift_pareto(frame: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic-only: mark support--lift non-dominated rows."""
    result = frame.copy()
    if result.empty:
        result["is_pareto_legacy"] = False
        return result
    points = result[["scenario_support", "lift"]].to_numpy(dtype=float)
    result["is_pareto_legacy"] = is_pareto_efficient(points)
    return result
