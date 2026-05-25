"""Affichage des tableaux de comparaison géométrique (notebook 01)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from metrics.geometry import GEOMETRY_METRIC_KEYS

from metrics.embedding_dims import embedding_dim_for_display_label

EMBEDDING_COMPARE_METHODS: tuple[str, ...] = (
    "raw_embedding",
    "batch_triplet",
    "supcon",
    "softtriple",
    "scgm_text",
)

# Sous-dossier de output_test/<corpus_id>/ pour l'embedding brut test
RAW_TEST_RESULTS_KEY = "raw_embedding"

METHOD_DISPLAY: dict[str, str] = {
    "raw_embedding": "Embedding brut",
    "scgm_text": "SCGM",
    "batch_triplet": "Batch Triplet",
    "softtriple": "SoftTriple",
    "supcon": "SupCon",
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

# Métriques val K-fold agrégées (μ±σ) pour le tableau BTP du notebook 01
KFOLD_SLIM_METRICS: tuple[str, ...] = (
    "eta2_macro_balanced",
    "eta2_macro_balanced_perc",
    "eta2_weighted",
    "rankme_global",
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


def default_embedding_dims_series(df: pd.DataFrame) -> pd.Series:
    """d par libellé méthode (1024 Qwen / contrastifs, 128 SCGM)."""
    if df.empty or "method" not in df.columns:
        return pd.Series(dtype=float)
    return pd.Series(
        [float(embedding_dim_for_display_label(m)) for m in df["method"].astype(str)],
        index=df.index,
        dtype=float,
    )


def fill_rankme_over_d(df: pd.DataFrame) -> pd.DataFrame:
    """rankme_over_d = rankme_global / d (1024 Qwen, 128 SCGM) si absent des CSV."""
    if df.empty or "rankme_global" not in df.columns:
        return df
    out = df.copy()
    rankme = pd.to_numeric(out["rankme_global"], errors="coerce")
    defaults = default_embedding_dims_series(out)
    if "embedding_dim" in out.columns:
        d = pd.to_numeric(out["embedding_dim"], errors="coerce")
        d = d.where(d.notna() & (d > 0), defaults)
    else:
        d = defaults
    out["embedding_dim"] = d
    d = d.replace(0, np.nan)
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


def format_mean_std(
    mean: Optional[float],
    std: Optional[float],
    *,
    decimals: int = 2,
) -> str:
    """Chaîne « m ± s » pour affichage notebook."""
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "—"
    m = float(mean)
    if std is None or (isinstance(std, float) and math.isnan(std)):
        return f"{m:.{decimals}f}"
    s = float(std)
    if s == 0.0:
        return f"{m:.{decimals}f}"
    return f"{m:.{decimals}f} ± {s:.{decimals}f}"


def _load_kfold_summary_row(method_dir: Path) -> Optional[dict]:
    path = method_dir / "metrics" / "kfold_summary.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def _enrich_kfold_rankme_over_d(df: pd.DataFrame) -> pd.DataFrame:
    """Dérive mean/std rankme_over_d à partir de rankme_global et d par méthode."""
    if df.empty or "mean_rankme_global" not in df.columns:
        return df
    out = df.copy()
    defaults = default_embedding_dims_series(out)
    if "embedding_dim" in out.columns:
        d = pd.to_numeric(out["embedding_dim"], errors="coerce")
        d = d.where(d.notna() & (d > 0), defaults)
    else:
        d = defaults
    out["embedding_dim"] = d
    d_safe = d.replace(0, np.nan)
    mean_rm = pd.to_numeric(out["mean_rankme_global"], errors="coerce")
    out["mean_rankme_over_d"] = mean_rm / d_safe
    if "std_rankme_global" in out.columns:
        std_rm = pd.to_numeric(out["std_rankme_global"], errors="coerce")
        out["std_rankme_over_d"] = std_rm / d_safe
    return out


def collect_kfold_btp_comparison(
    root: Path,
    method_keys: Sequence[str] = EMBEDDING_COMPARE_METHODS,
) -> pd.DataFrame:
    """
    Agrège metrics/kfold_summary.csv (validation K-fold) par méthode sous ``root/``.
    Colonnes : method, n_folds, mean_<metric>, std_<metric>.
    """
    rows: list[dict] = []
    for key in method_keys:
        method_dir = root / key
        if not method_dir.is_dir():
            continue
        raw = _load_kfold_summary_row(method_dir)
        if raw is None:
            continue
        entry: dict = {
            "method": method_label(key),
            "method_key": key,
            "n_folds": raw.get("n_folds"),
        }
        for metric in GEOMETRY_METRIC_KEYS:
            mean_col = f"mean_{metric}"
            std_col = f"std_{metric}"
            if mean_col in raw:
                entry[mean_col] = raw[mean_col]
            if std_col in raw:
                entry[std_col] = raw[std_col]
        rows.append(entry)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # η²_% manquant dans certains résumés K-fold
    if "mean_eta2_macro_balanced" in df.columns:
        eta2 = pd.to_numeric(df["mean_eta2_macro_balanced"], errors="coerce")
        if "mean_eta2_macro_balanced_perc" not in df.columns:
            df["mean_eta2_macro_balanced_perc"] = eta2 * 100.0
        else:
            perc = pd.to_numeric(df["mean_eta2_macro_balanced_perc"], errors="coerce")
            df["mean_eta2_macro_balanced_perc"] = perc.where(perc.notna(), eta2 * 100.0)
        if "std_eta2_macro_balanced_perc" not in df.columns and "std_eta2_macro_balanced" in df.columns:
            df["std_eta2_macro_balanced_perc"] = (
                pd.to_numeric(df["std_eta2_macro_balanced"], errors="coerce") * 100.0
            )
    return order_methods(_enrich_kfold_rankme_over_d(df))


def kfold_slim_metric_keys() -> tuple[str, ...]:
    """Clés affichées (inclut rankme_over_d si dérivé)."""
    keys = list(KFOLD_SLIM_METRICS)
    return tuple(keys)


def kfold_geometry_display_table(
    df_kfold: pd.DataFrame,
    *,
    metrics: Sequence[str] = KFOLD_SLIM_METRICS,
    decimals: int = 2,
) -> pd.DataFrame:
    """Tableau lisible μ±σ pour le notebook (une colonne par métrique)."""
    if df_kfold.empty:
        return df_kfold
    df = _enrich_kfold_rankme_over_d(df_kfold)
    display_metrics = list(metrics)
    if "rankme_global" in display_metrics and "mean_rankme_over_d" in df.columns:
        display_metrics = [
            "rankme_over_d" if m == "rankme_global" else m for m in display_metrics
        ]
    out = pd.DataFrame({"method": df["method"].astype(str)})
    if "n_folds" in df.columns:
        out["n_folds"] = df["n_folds"]
    for metric in display_metrics:
        mean_col = f"mean_{metric}"
        std_col = f"std_{metric}"
        if mean_col not in df.columns:
            continue
        out[metric] = [
            format_mean_std(
                row.get(mean_col),
                row.get(std_col) if std_col in df.columns else None,
                decimals=decimals,
            )
            for _, row in df.iterrows()
        ]
    return out


def kfold_barplot_frame(
    df_kfold: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """DataFrame method / mean / std pour barres d'erreur matplotlib."""
    if df_kfold.empty:
        return pd.DataFrame(columns=["method", "mean", "std"])
    df = _enrich_kfold_rankme_over_d(df_kfold)
    key = metric
    if metric == "rankme_over_d" and f"mean_{metric}" not in df.columns:
        key = "rankme_global"
    mean_col = f"mean_{key}"
    std_col = f"std_{key}"
    if mean_col not in df.columns:
        return pd.DataFrame(columns=["method", "mean", "std"])
    return pd.DataFrame(
        {
            "method": df["method"].astype(str),
            "mean": pd.to_numeric(df[mean_col], errors="coerce"),
            "std": pd.to_numeric(df.get(std_col, 0.0), errors="coerce").fillna(0.0),
        }
    )
