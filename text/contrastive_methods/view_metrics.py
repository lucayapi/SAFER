"""Tableau unifié de métriques pour les notebooks de vue contrastif."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from safer_core.classification_eval import CLASSIFICATION_METRIC_KEYS

__all__ = [
    "build_view_classification_summary_table",
    "format_ood_summary_line",
    "validate_contrastive_results_dir",
]

_METRIC_COLS = ("balanced_accuracy", "macro_f1", "accuracy")
_CV_METRIC_MAP = {
    "balanced_accuracy": ("mean_val_balanced_accuracy", "std_val_balanced_accuracy"),
    "macro_f1": ("mean_val_macro_f1", "std_val_macro_f1"),
    "accuracy": ("mean_val_accuracy", "std_val_accuracy"),
}
_CV_METRIC_MAP_ALT = {
    "balanced_accuracy": ("mean_balanced_accuracy", "std_balanced_accuracy"),
    "macro_f1": ("mean_macro_f1", "std_macro_f1"),
    "accuracy": ("mean_accuracy", "std_accuracy"),
}


def _read_optional_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    return pd.read_csv(path)


def _format_pm(mean: Any, std: Any) -> str:
    try:
        m = float(mean)
        s = float(std) if std is not None and not (isinstance(std, float) and np.isnan(std)) else 0.0
        if not np.isfinite(m):
            return "—"
        return f"{m:.3f} ± {s:.3f}"
    except (TypeError, ValueError):
        return "—"


def _format_scalar(value: Any) -> str:
    try:
        x = float(value)
        if not np.isfinite(x):
            return "—"
        return f"{x:.3f}"
    except (TypeError, ValueError):
        return "—"


def validate_contrastive_results_dir(results_dir: str | Path) -> Path:
    """Vérifie qu'un dossier ressemble à un run contrastif."""
    root = Path(results_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dossier absent : {root}")
    markers = (
        root / "metrics" / "kfold_summary.csv",
        root / "metrics" / "metrics_classification_btp.csv",
        root / "metrics" / "cross_domain_generalization.csv",
        root / "checkpoints" / "best_model" / "config.json",
        root / "embeddings" / "projected_btp.npy",
    )
    if not any(p.is_file() for p in markers):
        raise FileNotFoundError(
            f"Dossier non reconnu comme run contrastif : {root}\n"
            "Attendu : metrics/kfold_summary.csv, metrics_classification_btp.csv, "
            "checkpoints/best_model ou embeddings/projected_btp.npy"
        )
    return root


def _cv_row_from_summary(kfold_df: pd.DataFrame) -> dict[str, Any]:
    if kfold_df.empty:
        return {}
    row = kfold_df.iloc[0]
    out: dict[str, Any] = {"phase": "cv_val", "corpus": "btp"}
    for metric in _METRIC_COLS:
        found = False
        for mapping in (_CV_METRIC_MAP, _CV_METRIC_MAP_ALT):
            keys = mapping.get(metric)
            if not keys:
                continue
            mean_key, std_key = keys
            if mean_key in row.index:
                out[metric] = _format_pm(row.get(mean_key), row.get(std_key))
                found = True
                break
        if not found:
            out[metric] = "—"
    return out


def _lr_row_from_classification(df: pd.DataFrame, corpus: str) -> dict[str, Any]:
    if df is None or df.empty:
        return {}
    row = df.iloc[0]
    out: dict[str, Any] = {"phase": "lr_eval", "corpus": corpus}
    for metric in _METRIC_COLS:
        out[metric] = _format_scalar(row.get(metric))
    return out


def build_view_classification_summary_table(
    results_dir: str | Path,
    *,
    test_corpora: Sequence[str],
) -> pd.DataFrame:
    """
    Tableau unique : CV BTP (μ±σ) + évaluation LR BTP + corpus OOD.

    Colonnes : phase, corpus, balanced_accuracy, macro_f1, accuracy.
    """
    root = Path(results_dir).resolve()
    metrics_dir = root / "metrics"
    rows: list[dict[str, Any]] = []

    kfold = _read_optional_csv(metrics_dir / "kfold_summary.csv")
    if kfold is not None:
        cv_row = _cv_row_from_summary(kfold)
        if cv_row:
            rows.append(cv_row)

    btp_cls = _read_optional_csv(metrics_dir / "metrics_classification_btp.csv")
    if btp_cls is not None:
        lr_btp = _lr_row_from_classification(btp_cls, "btp")
        if lr_btp:
            rows.append(lr_btp)

    for corpus_id in test_corpora:
        test_cls = _read_optional_csv(metrics_dir / f"metrics_classification_test_{corpus_id}.csv")
        if test_cls is not None:
            lr_row = _lr_row_from_classification(test_cls, str(corpus_id))
            if lr_row:
                rows.append(lr_row)

    if not rows:
        return pd.DataFrame(columns=["phase", "corpus", *_METRIC_COLS])
    return pd.DataFrame(rows)[["phase", "corpus", *_METRIC_COLS]]


def format_ood_summary_line(results_dir: str | Path) -> Optional[str]:
    """Ligne texte BA moyenne / pire corpus OOD."""
    root = Path(results_dir).resolve()
    cross = _read_optional_csv(root / "metrics" / "cross_domain_generalization.csv")
    if cross is not None and not cross.empty:
        row = cross.iloc[0]
        ba_avg = row.get("ba_ood_avg")
        ba_worst = row.get("ba_ood_worst")
        if pd.notna(ba_avg) or pd.notna(ba_worst):
            parts = []
            if pd.notna(ba_avg):
                parts.append(f"BA moyenne : {float(ba_avg):.3f}")
            if pd.notna(ba_worst):
                parts.append(f"BA pire corpus : {float(ba_worst):.3f}")
            return "OOD — " + " | ".join(parts)

    all_test = _read_optional_csv(root / "metrics" / "all_test_corpora_metrics.csv")
    if all_test is not None and not all_test.empty and "balanced_accuracy" in all_test.columns:
        vals = all_test["balanced_accuracy"].astype(float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            return (
                f"OOD — BA moyenne : {float(vals.mean()):.3f} | "
                f"BA pire corpus : {float(vals.min()):.3f}"
            )
    return None


def build_macro_ft_classification_summary_table(
    cv_summary: Optional[pd.DataFrame],
    metrics_by_corpus: Mapping[str, pd.DataFrame],
    *,
    test_corpora: Sequence[str],
) -> pd.DataFrame:
    """Tableau unifié pour supervised_macro_ft (même schéma que contrastif)."""
    rows: list[dict[str, Any]] = []

    if cv_summary is not None and not cv_summary.empty:
        cv_row = _cv_row_from_summary(cv_summary)
        if cv_row:
            rows.append(cv_row)

    for corpus_id in ("btp", *test_corpora):
        mdf = metrics_by_corpus.get(corpus_id)
        if mdf is not None and not mdf.empty:
            lr_row = _lr_row_from_classification(mdf, corpus_id)
            if lr_row:
                rows.append(lr_row)

    if not rows:
        return pd.DataFrame(columns=["phase", "corpus", *_METRIC_COLS])
    return pd.DataFrame(rows)[["phase", "corpus", *_METRIC_COLS]]
