"""Sous-échantillonnage accidents / unités factuelles."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from annotation.config import IntOrAll


def sample_accidents_and_units(
    df: pd.DataFrame,
    *,
    n_accidents: IntOrAll = "all",
    units_per_accident: IntOrAll = "all",
    seed: int = 42,
    accident_col: str = "accident_id",
    accident_sample_frac: Optional[float] = None,
) -> pd.DataFrame:
    if accident_col not in df.columns:
        raise ValueError(f"Colonne {accident_col!r} absente du DataFrame.")

    rng = np.random.RandomState(int(seed))
    accident_ids = df[accident_col].astype(str).unique().tolist()

    if accident_sample_frac is not None:
        frac = float(accident_sample_frac)
        if not (0.0 < frac <= 1.0):
            raise ValueError(
                f"accident_sample_frac doit être dans (0, 1], reçu : {frac!r}"
            )
        n = max(1, int(round(frac * len(accident_ids))))
        n = min(n, len(accident_ids))
        accident_ids = rng.choice(accident_ids, size=n, replace=False).tolist()
    elif n_accidents != "all":
        n = min(int(n_accidents), len(accident_ids))
        accident_ids = rng.choice(accident_ids, size=n, replace=False).tolist()

    parts: list[pd.DataFrame] = []
    for accident_id in accident_ids:
        group = df[df[accident_col].astype(str) == str(accident_id)].copy()
        if units_per_accident != "all":
            k = min(int(units_per_accident), len(group))
            group = group.sample(n=k, random_state=rng)
        parts.append(group)

    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def sampling_stats(df: pd.DataFrame, *, accident_col: str = "accident_id") -> dict[str, int]:
    return {
        "n_rows": int(len(df)),
        "n_accidents": int(df[accident_col].nunique()) if accident_col in df.columns else 0,
        "estimated_api_calls": int(len(df)),
    }
