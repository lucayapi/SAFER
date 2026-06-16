"""Intra-role Preservation Ratio (IPR) from geometry metrics (T, W_r)."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from metrics.embedding_geometry_separation import MACRO_NAMES

DEFAULT_S_COL = "T_macro_balanced"
DEFAULT_BASELINE_LABEL = "Embedding brut"
IPR_ROLE_COLUMNS: tuple[str, ...] = tuple(f"IPR_{r}" for r in MACRO_NAMES)
IPR_MEAN_COLUMN = "IPR_mean"
IPR_COLUMNS: tuple[str, ...] = IPR_ROLE_COLUMNS + (IPR_MEAN_COLUMN,)


def _finite_positive(value: Any, eps: float) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x) or x <= eps:
        return None
    return x


def rho_r(
    row: Mapping[str, Any],
    *,
    s_col: str = DEFAULT_S_COL,
    roles: Sequence[str] = MACRO_NAMES,
    eps: float = 1e-12,
) -> dict[str, float]:
    """ρ_r = S / W_r with S = T_macro_balanced (global dispersion)."""
    s = _finite_positive(row.get(s_col), eps)
    out: dict[str, float] = {}
    for role in roles:
        w_col = f"W_{role}"
        w = _finite_positive(row.get(w_col), eps)
        if s is None or w is None:
            out[role] = float("nan")
        else:
            out[role] = s / w
    return out


def ipr_r_from_rho(
    rho_baseline: Mapping[str, float],
    rho_method: Mapping[str, float],
    *,
    roles: Sequence[str] = MACRO_NAMES,
) -> dict[str, float]:
    """IPR_r = ρ_r(brut) / ρ_r(m)."""
    out: dict[str, float] = {}
    for role in roles:
        rb = rho_baseline.get(role, float("nan"))
        rm = rho_method.get(role, float("nan"))
        if not np.isfinite(rb) or not np.isfinite(rm) or rm == 0.0:
            out[role] = float("nan")
        else:
            out[role] = rb / rm
    return out


def ipr_mean(ipr_by_role: Mapping[str, float], *, roles: Sequence[str] = MACRO_NAMES) -> float:
    """Arithmetic mean over roles with finite IPR_r."""
    vals = [
        float(ipr_by_role[r])
        for r in roles
        if r in ipr_by_role and np.isfinite(ipr_by_role[r])
    ]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def compute_ipr_from_geometry_rows(
    baseline_geom: Mapping[str, Any],
    method_geom: Mapping[str, Any],
    *,
    s_col: str = DEFAULT_S_COL,
    roles: Sequence[str] = MACRO_NAMES,
    eps: float = 1e-12,
) -> dict[str, float]:
    """IPR_* from two geometry dicts (e.g. raw val vs fine-tuned val on same fold)."""
    rho_base = rho_r(baseline_geom, s_col=s_col, roles=roles, eps=eps)
    rho_m = rho_r(method_geom, s_col=s_col, roles=roles, eps=eps)
    ipr = ipr_r_from_rho(rho_base, rho_m, roles=roles)
    out = {f"IPR_{role}": ipr[role] for role in roles}
    out[IPR_MEAN_COLUMN] = ipr_mean(ipr, roles=roles)
    return out


def _baseline_row(
    df: pd.DataFrame,
    *,
    baseline_label: str,
) -> Optional[pd.Series]:
    if df.empty or "method" not in df.columns:
        return None
    mask = df["method"].astype(str) == baseline_label
    if not mask.any():
        return None
    return df.loc[mask].iloc[0]


def compute_ipr_columns(
    df: pd.DataFrame,
    *,
    baseline_label: Optional[str] = None,
    s_col: str = DEFAULT_S_COL,
    roles: Sequence[str] = MACRO_NAMES,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Add IPR_A0…IPR_C and IPR_mean vs embedding brut (same DataFrame / corpus).

    Requires T_macro_balanced and W_A0…W_C on each row.
    """
    if df.empty:
        out = df.copy()
        for col in IPR_COLUMNS:
            out[col] = pd.Series(dtype=float)
        return out

    baseline = baseline_label or DEFAULT_BASELINE_LABEL
    out = df.copy()
    base = _baseline_row(out, baseline_label=baseline)
    rho_base = rho_r(base, s_col=s_col, roles=roles, eps=eps) if base is not None else None

    for col in IPR_COLUMNS:
        out[col] = float("nan")

    if rho_base is None:
        return out

    for idx, row in out.iterrows():
        rho_m = rho_r(row, s_col=s_col, roles=roles, eps=eps)
        ipr = ipr_r_from_rho(rho_base, rho_m, roles=roles)
        for role in roles:
            out.at[idx, f"IPR_{role}"] = ipr[role]
        out.at[idx, IPR_MEAN_COLUMN] = ipr_mean(ipr, roles=roles)

    return out


def ipr_display_table(
    df: pd.DataFrame,
    *,
    decimals: int = 3,
) -> pd.DataFrame:
    """method + IPR_mean + IPR per role (rounded)."""
    if df.empty:
        return pd.DataFrame(columns=["method", IPR_MEAN_COLUMN, *IPR_ROLE_COLUMNS])
    cols = ["method"] + [c for c in [IPR_MEAN_COLUMN, *IPR_ROLE_COLUMNS] if c in df.columns]
    if "method" not in df.columns:
        return pd.DataFrame()
    table = df[cols].copy()
    for col in IPR_COLUMNS:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce").round(decimals)
    return table
