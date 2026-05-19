"""Affichage des tableaux de comparaison géométrique (notebook 01)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from metrics.embedding_dims import embedding_dim_for_display_label

EMBEDDING_COMPARE_METHODS: tuple[str, ...] = (
    "raw_embedding",
    "batch_triplet",
    "supcon",
    "softtriple",
    "scgm_text",
)

# Dossier resultats/ pour les métriques raw sur le corpus test (métallurgie)
RAW_TEST_RESULTS_KEY = "raw_embedding_test"

METHOD_DISPLAY: dict[str, str] = {
    "raw_embedding": "Embedding brut",
    "scgm_text": "SCGM",
    "batch_triplet": "Batch Triplet",
    "softtriple": "SoftTriple",
    "supcon": "SupCon",
    "malt": "MALT",
}

METHOD_DISPLAY_TO_KEY: dict[str, str] = {v: k for k, v in METHOD_DISPLAY.items()}

GEOM_DISPLAY_COLS: tuple[str, ...] = (
    "method",
    "eta2_macro_balanced",
    "eta2_macro_balanced_perc",
    "eta2_weighted",
    "rankme_over_d",
    "embedding_dim",
    "c1_global",
    "c10_global",
)


def method_label(method_key: str) -> str:
    return METHOD_DISPLAY.get(method_key, method_key)


def normalize_method_display_name(name: str, method_key: str) -> str:
    """Libellé court sans suffixe _btp / _test ; repli sur METHOD_DISPLAY."""
    s = str(name).strip()
    for suffix in ("_btp", "_test", "_BTP", "_TEST"):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    if s in METHOD_DISPLAY.values():
        return s
    if method_key in METHOD_DISPLAY:
        return method_label(method_key)
    return s


def fill_eta2_macro_balanced_perc(df: pd.DataFrame) -> pd.DataFrame:
    """η²_% = 100 × η² macro balancé lorsque la colonne est absente ou NaN."""
    if df.empty or "eta2_macro_balanced" not in df.columns:
        return df
    out = df.copy()
    eta2 = pd.to_numeric(out["eta2_macro_balanced"], errors="coerce")
    if "eta2_macro_balanced_perc" not in out.columns:
        out["eta2_macro_balanced_perc"] = eta2 * 100.0
    else:
        perc = pd.to_numeric(out["eta2_macro_balanced_perc"], errors="coerce")
        out["eta2_macro_balanced_perc"] = perc.where(perc.notna(), eta2 * 100.0)
    if "delta_macro_pct" in out.columns and out["eta2_macro_balanced_perc"].isna().any():
        legacy = pd.to_numeric(out["delta_macro_pct"], errors="coerce")
        out["eta2_macro_balanced_perc"] = out["eta2_macro_balanced_perc"].where(
            out["eta2_macro_balanced_perc"].notna(), legacy
        )
    return out


def fill_rankme_over_d(df: pd.DataFrame) -> pd.DataFrame:
    """rankme_over_d = rankme_global / d (1024 Qwen, 128 SCGM) si absent des CSV."""
    if df.empty or "rankme_global" not in df.columns:
        return df
    out = df.copy()
    rankme = pd.to_numeric(out["rankme_global"], errors="coerce")
    if "embedding_dim" in out.columns:
        d = pd.to_numeric(out["embedding_dim"], errors="coerce")
    else:
        d = pd.Series(
            [embedding_dim_for_display_label(m) for m in out["method"].astype(str)],
            index=out.index,
            dtype=float,
        )
        out["embedding_dim"] = d
    d = pd.to_numeric(out["embedding_dim"], errors="coerce").replace(0, np.nan)
    computed = rankme / d
    if "rankme_over_d" not in out.columns:
        out["rankme_over_d"] = computed
    else:
        over = pd.to_numeric(out["rankme_over_d"], errors="coerce")
        out["rankme_over_d"] = over.where(over.notna(), computed)
    return out


def order_methods(
    df: pd.DataFrame,
    keys: Sequence[str] = EMBEDDING_COMPARE_METHODS,
) -> pd.DataFrame:
    """Ordonne les lignes selon keys ; lignes hors liste en fin."""
    if df.empty or "method" not in df.columns:
        return df
    label_order = [METHOD_DISPLAY.get(k, k) for k in keys]
    extra = [m for m in df["method"].astype(str).unique() if m not in label_order]
    order = label_order + extra
    cat = pd.Categorical(df["method"].astype(str), categories=order, ordered=True)
    out = df.copy()
    out["_sort"] = cat
    out = out.sort_values("_sort").drop(columns="_sort")
    return out.reset_index(drop=True)


def slim_geometry_table(df: pd.DataFrame) -> pd.DataFrame:
    slim = fill_rankme_over_d(fill_eta2_macro_balanced_perc(df))
    cols = [c for c in GEOM_DISPLAY_COLS if c in slim.columns]
    if not cols:
        return slim
    return slim[cols].copy()
