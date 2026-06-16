"""Affichage des tableaux de comparaison géométrique (notebook 01)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from metrics.geometry import GEOMETRY_METRIC_KEYS
from metrics.embedding_geometry_separation import MACRO_NAMES
from metrics.intra_role_preservation import (
    DEFAULT_BASELINE_LABEL,
    IPR_COLUMNS,
    IPR_MEAN_COLUMN,
    compute_ipr_columns,
    ipr_display_table,
)

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
    "embedding_dim",
)

KFOLD_IPR_METRICS: tuple[str, ...] = IPR_COLUMNS

# Métriques val K-fold agrégées (μ±σ) pour le tableau BTP du notebook 01
KFOLD_SLIM_METRICS: tuple[str, ...] = (
    "eta2_macro_balanced",
    "eta2_macro_balanced_perc",
    "eta2_weighted",
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
    slim = fill_eta2_macro_balanced_perc(df)
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
        for metric in IPR_COLUMNS:
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
    return order_methods(df)


def kfold_slim_metric_keys() -> tuple[str, ...]:
    return tuple(KFOLD_SLIM_METRICS)


def kfold_geometry_display_table(
    df_kfold: pd.DataFrame,
    *,
    metrics: Sequence[str] = KFOLD_SLIM_METRICS,
    decimals: int = 2,
) -> pd.DataFrame:
    """Tableau lisible μ±σ pour le notebook (une colonne par métrique)."""
    if df_kfold.empty:
        return df_kfold
    df = df_kfold
    display_metrics = list(metrics)
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


def kfold_ipr_display_table(
    df_kfold: pd.DataFrame,
    *,
    decimals: int = 3,
) -> pd.DataFrame:
    """μ±σ IPR_mean et IPR par rôle (validation K-fold)."""
    return kfold_geometry_display_table(
        df_kfold,
        metrics=KFOLD_IPR_METRICS,
        decimals=decimals,
    )


def kfold_barplot_frame(
    df_kfold: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """DataFrame method / mean / std pour barres d'erreur matplotlib."""
    if df_kfold.empty:
        return pd.DataFrame(columns=["method", "mean", "std"])
    df = df_kfold
    mean_col = f"mean_{metric}"
    std_col = f"std_{metric}"
    if mean_col not in df.columns:
        return pd.DataFrame(columns=["method", "mean", "std"])
    return pd.DataFrame(
        {
            "method": df["method"].astype(str),
            "mean": pd.to_numeric(df[mean_col], errors="coerce"),
            "std": pd.to_numeric(df.get(std_col, 0.0), errors="coerce").fillna(0.0),
        }
    )


def enrich_geometry_with_ipr(
    df: pd.DataFrame,
    *,
    baseline_label: str | None = None,
) -> pd.DataFrame:
    """Ajoute IPR_* vs embedding brut et réordonne les méthodes."""
    baseline = baseline_label or METHOD_DISPLAY.get("raw_embedding", DEFAULT_BASELINE_LABEL)
    enriched = compute_ipr_columns(df, baseline_label=baseline)
    return order_methods(enriched)


def joint_eta2_ipr_table(df: pd.DataFrame, *, decimals: int = 3) -> pd.DataFrame:
    """η² macro balancé (%) + IPR_mean pour lecture conjointe."""
    if df.empty or "method" not in df.columns:
        return pd.DataFrame()
    work = enrich_geometry_with_ipr(df) if IPR_MEAN_COLUMN not in df.columns else order_methods(df)
    work = fill_eta2_macro_balanced_perc(work)
    cols = ["method"]
    if "eta2_macro_balanced_perc" in work.columns:
        cols.append("eta2_macro_balanced_perc")
    elif "eta2_macro_balanced" in work.columns:
        cols.append("eta2_macro_balanced")
    if IPR_MEAN_COLUMN in work.columns:
        cols.append(IPR_MEAN_COLUMN)
    out = work[cols].copy()
    if "eta2_macro_balanced_perc" in out.columns:
        out["eta2_macro_balanced_perc"] = pd.to_numeric(
            out["eta2_macro_balanced_perc"], errors="coerce"
        ).round(decimals)
    if IPR_MEAN_COLUMN in out.columns:
        out[IPR_MEAN_COLUMN] = pd.to_numeric(out[IPR_MEAN_COLUMN], errors="coerce").round(decimals)
    return out


def plot_ipr_comparison(
    df: pd.DataFrame,
    title_prefix: str,
    *,
    figsize: tuple[float, float] = (10.0, 4.0),
) -> None:
    """Barres groupées IPR par rôle + IPR_mean ; ligne y=1 = référence brut."""
    import matplotlib.pyplot as plt

    work = enrich_geometry_with_ipr(df) if IPR_MEAN_COLUMN not in df.columns else order_methods(df)
    if work.empty or IPR_MEAN_COLUMN not in work.columns:
        return

    methods = work["method"].astype(str).tolist()
    series_labels = list(MACRO_NAMES) + ["moy."]
    series_cols = [f"IPR_{r}" for r in MACRO_NAMES] + [IPR_MEAN_COLUMN]
    n_methods = len(methods)
    n_series = len(series_labels)
    x = np.arange(n_methods)
    width = 0.8 / max(n_series, 1)

    fig, ax = plt.subplots(figsize=figsize)
    for i, (label, col) in enumerate(zip(series_labels, series_cols)):
        if col not in work.columns:
            continue
        offset = (i - (n_series - 1) / 2.0) * width
        vals = pd.to_numeric(work[col], errors="coerce").astype(float).tolist()
        ax.bar(x + offset, vals, width=width * 0.95, label=label)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="référence brut (IPR=1)")
    ax.set_ylabel("IPR (ρ brut / ρ méthode)")
    ax.set_title(f"{title_prefix} — préservation intra-rôle (IPR)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    plt.show()


def plot_eta2_ipr_dual(
    df: pd.DataFrame,
    title_prefix: str,
    *,
    figsize: tuple[float, float] = (12.0, 4.0),
) -> None:
    """Deux panneaux : η² macro balancé (%) et IPR_mean."""
    import matplotlib.pyplot as plt

    work = enrich_geometry_with_ipr(df) if IPR_MEAN_COLUMN not in df.columns else order_methods(df)
    if work.empty:
        return
    work = fill_eta2_macro_balanced_perc(work)
    methods = work["method"].astype(str).tolist()
    x = np.arange(len(methods))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if "eta2_macro_balanced_perc" in work.columns:
        eta2 = pd.to_numeric(work["eta2_macro_balanced_perc"], errors="coerce")
        axes[0].bar(x, eta2.astype(float))
        axes[0].set_ylabel("η² macro balancé (%)")
        axes[0].set_title(f"{title_prefix} — séparation macro")
    else:
        axes[0].set_visible(False)

    if IPR_MEAN_COLUMN in work.columns:
        ipr = pd.to_numeric(work[IPR_MEAN_COLUMN], errors="coerce")
        axes[1].bar(x, ipr.astype(float))
        axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
        axes[1].set_ylabel("IPR moyen")
        axes[1].set_title(f"{title_prefix} — préservation intra-rôle")
    else:
        axes[1].set_visible(False)

    for ax in axes:
        if ax.get_visible():
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=30, ha="right")

    fig.suptitle(f"{title_prefix} — η² et IPR (lecture conjointe)", y=1.02)
    fig.tight_layout()
    plt.show()
