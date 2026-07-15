"""Comparaison géométrique η² entre méthodes et corpus (notebook 09)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from metrics.embedding_geometry_separation import (
    METRICS_TABLE_COLUMNS,
    PRIMARY_SELECTION_METRIC,
    build_geometry_metrics_row,
)
from safer_core.brand_style import (
    SERIES_PALETTE,
    apply_matplotlib_brand,
    bar_kwargs,
    style_axes,
    style_figure,
)
from safer_core.paths import TEXT_ROOT, resolve_repo_path
from safer_core.test_corpus import resolve_projected_embeddings_in_dir, resolve_test_corpus

SUMMARY_DISPLAY_COLUMNS: Tuple[str, ...] = (
    "method",
    "corpus",
    "eta2_macro_balanced_perc",
    "eta2_macro_balanced",
    "eta2_weighted",
    "embedding_dim",
    "n_A0",
    "n_A1",
    "n_B",
    "n_C",
    "macros_ignored",
)

MethodSpec = Dict[str, Any]


def _resolve_results_dir(results_dir: Union[str, Path, None], *, anchor: Path) -> Path:
    if results_dir is None:
        raise ValueError("results_dir requis pour kind='projected'")
    return resolve_repo_path(str(results_dir), repo_root=anchor)


def load_raw_qwen_embeddings(
    corpus_id: str,
    *,
    label_col: str = "pred_label",
    anchor: Optional[Path] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Charge embeddings Qwen bruts (CSV) + métadonnées pour un corpus."""
    from scgm_text.dataset_text_embeddings import merge_metadata_with_embeddings
    from scgm_text.utils_io import create_doc_id_if_missing

    root = Path(anchor or TEXT_ROOT).resolve()
    spec = resolve_test_corpus(corpus_id, anchor=root)
    meta = create_doc_id_if_missing(pd.read_csv(spec.data_csv))
    slim = meta.drop(columns=[c for c in meta.columns if c.startswith("dim_")], errors="ignore")
    merged, dim_cols = merge_metadata_with_embeddings(
        slim, str(spec.emb_csv), strict=False, corpus_id=corpus_id
    )
    if label_col not in merged.columns:
        raise KeyError(f"Colonne {label_col!r} absente pour corpus {corpus_id}")
    x = merged[dim_cols].to_numpy(dtype=np.float64)
    return x, merged


def load_projected_embeddings(
    results_dir: Union[str, Path],
    corpus_id: str,
    *,
    method_key: Optional[str] = None,
    label_col: str = "pred_label",
    anchor: Optional[Path] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Charge ``projected_<corpus>.npy`` + metadata depuis un dossier de run."""
    root = Path(anchor or TEXT_ROOT).resolve()
    run_dir = _resolve_results_dir(results_dir, anchor=root)
    pair = resolve_projected_embeddings_in_dir(
        run_dir, corpus_id, method=method_key, anchor=root
    )
    if pair is None:
        raise FileNotFoundError(
            f"Embeddings projetés absents pour {corpus_id} sous {run_dir / 'embeddings'}"
        )
    projected = np.load(pair[0])
    meta = pd.read_csv(pair[1])
    if label_col not in meta.columns:
        raise KeyError(f"Colonne {label_col!r} absente dans {pair[1]}")
    n = min(len(meta), projected.shape[0])
    return projected[:n], meta.iloc[:n].copy()


def _align_labels(meta: pd.DataFrame, label_col: str) -> np.ndarray:
    return meta[label_col].astype(str).to_numpy()


def compute_geometry_for_method_corpus(
    spec: MethodSpec,
    corpus_id: str,
    *,
    label_col: str = "pred_label",
    l2_normalize: bool = False,
    anchor: Optional[Path] = None,
) -> Dict[str, Any]:
    """Calcule une ligne de métriques η² pour (méthode, corpus)."""
    name = str(spec.get("name") or spec.get("method") or "unknown")
    kind = str(spec.get("kind", "projected")).strip().lower()
    root = Path(anchor or TEXT_ROOT).resolve()

    if kind == "raw":
        x, meta = load_raw_qwen_embeddings(corpus_id, label_col=label_col, anchor=root)
    elif kind == "projected":
        x, meta = load_projected_embeddings(
            spec["results_dir"],
            corpus_id,
            method_key=spec.get("method_key"),
            label_col=label_col,
            anchor=root,
        )
    else:
        raise ValueError(f"kind non supporté : {kind!r} (attendu raw | projected)")

    y = _align_labels(meta, label_col)
    row = build_geometry_metrics_row(
        x,
        y,
        method=name,
        l2_normalize=l2_normalize,
        embedding_dim=int(x.shape[1]),
    )
    row["corpus"] = str(corpus_id)
    return row


def build_geometry_comparison_table(
    method_specs: Sequence[MethodSpec],
    corpora: Sequence[str],
    *,
    label_col: str = "pred_label",
    l2_normalize: bool = False,
    anchor: Optional[Path] = None,
    skip_errors: bool = False,
) -> pd.DataFrame:
    """
    Tableau long : une ligne par (méthode, corpus) avec métriques η².

    Si ``skip_errors=True``, les combinaisons en échec sont omises avec un warning.
    """
    import warnings

    rows: List[Dict[str, Any]] = []
    for spec in method_specs:
        for corpus_id in corpora:
            try:
                rows.append(
                    compute_geometry_for_method_corpus(
                        spec,
                        corpus_id,
                        label_col=label_col,
                        l2_normalize=l2_normalize,
                        anchor=anchor,
                    )
                )
            except (FileNotFoundError, KeyError, ValueError) as exc:
                if not skip_errors:
                    raise
                warnings.warn(
                    f"{spec.get('name')} / {corpus_id} : {exc}",
                    UserWarning,
                    stacklevel=2,
                )
    if not rows:
        return pd.DataFrame(columns=list(SUMMARY_DISPLAY_COLUMNS))
    df = pd.DataFrame(rows)
    cols = [c for c in SUMMARY_DISPLAY_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in cols and c in METRICS_TABLE_COLUMNS]
    return df[cols + extra]


def plot_geometry_comparison_bars(
    summary: pd.DataFrame,
    corpus_id: str,
    *,
    metric: str = PRIMARY_SELECTION_METRIC,
    fig_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None,
    show: bool = True,
) -> Optional[Path]:
    """Barplot η² par méthode pour un corpus donné."""
    import matplotlib.pyplot as plt

    if summary.empty or metric not in summary.columns:
        return None

    sub = summary.loc[summary["corpus"].astype(str) == str(corpus_id)].copy()
    if sub.empty:
        return None

    sub = sub.sort_values(metric, ascending=False)
    methods = sub["method"].astype(str).tolist()
    values = sub[metric].astype(float).to_numpy()
    colors = [SERIES_PALETTE[i % len(SERIES_PALETTE)] for i in range(len(methods))]

    apply_matplotlib_brand()
    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.1), 5))
    style_figure(fig)
    x = np.arange(len(methods))
    bk = bar_kwargs()
    ax.bar(x, values, color=colors, edgecolor=bk["edgecolor"], linewidth=bk["linewidth"])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    style_axes(
        ax,
        title=f"η² macro balanced (%) — {corpus_id}",
        title_level="h3",
    )
    ax.set_ylabel("η² macro balanced (%)")
    ax.set_xlabel("")
    ymax = float(np.nanmax(values)) if values.size else 100.0
    ax.set_ylim(0, max(100.0, ymax * 1.05))
    plt.tight_layout()

    out_path: Optional[Path] = None
    if fig_dir is not None:
        out = Path(fig_dir)
        out.mkdir(parents=True, exist_ok=True)
        fname = filename or f"geometry_eta2_{corpus_id}.png"
        out_path = out / fname
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


__all__ = [
    "SUMMARY_DISPLAY_COLUMNS",
    "load_raw_qwen_embeddings",
    "load_projected_embeddings",
    "compute_geometry_for_method_corpus",
    "build_geometry_comparison_table",
    "plot_geometry_comparison_bars",
]
