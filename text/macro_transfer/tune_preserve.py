"""Tuning ``loss_weights.preserve`` (λ_pres) pour macro_transfer TPN — métriques agrégées."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from macro_transfer.constants import MACRO_NAMES
from macro_transfer.report_tables import load_transfer_metrics_pair

# Grille par défaut (λ_pres = loss_weights.preserve)
DEFAULT_LAMBDA_PRES_GRID: tuple[float, ...] = (0.0, 0.05, 0.10, 0.25, 0.50)

ALL_TPN_ENCODERS: tuple[str, ...] = (
    "scgm_text",
    "softtriple",
    "supcon",
    "batch_triplet",
)


def lambda_pres_tag(value: float) -> str:
    """Nom de dossier stable pour une valeur de λ_pres."""
    s = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def lambda_pres_run_dir(tune_root: Path, lambda_pres: float) -> Path:
    return Path(tune_root) / f"lambda_{lambda_pres_tag(lambda_pres)}"


def shared_projected_cache_dir(method_root: Path) -> Path:
    """Cache phase 1 partagé entre toutes les valeurs de λ_pres pour un encodeur."""
    return Path(method_root) / "tune_preserve" / "_projected_cache"


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def aggregate_topic_metrics_from_stats(
    macro_stats: pd.DataFrame,
    *,
    weight_col: str = "n_units",
) -> Dict[str, Any]:
    """
    Agrège R_m, K_m, r_noise depuis ``summary/macro_topic_stats.csv``.

    Mapping par macro :
      R_m = plus_gros_topic_pct / 100
      K_m = n_topics
      r_noise = bruit_pct / 100

    Retourne aussi les moyennes pondérées globales (par ``n_units``).
    """
    if macro_stats.empty:
        return {
            "R_m": float("nan"),
            "K_m": float("nan"),
            "r_noise": float("nan"),
        }

    df = macro_stats.copy()
    for col in ("plus_gros_topic_pct", "n_topics", "bruit_pct", weight_col):
        if col not in df.columns:
            raise ValueError(f"macro_topic_stats.csv : colonne {col!r} absente")

    df["R_m"] = df["plus_gros_topic_pct"].astype(float) / 100.0
    df["K_m"] = df["n_topics"].astype(float)
    df["r_noise"] = df["bruit_pct"].astype(float) / 100.0

    weights = df[weight_col].astype(float).to_numpy()
    w_sum = float(weights.sum())
    if w_sum <= 0:
        w_norm = np.ones(len(df), dtype=np.float64) / max(len(df), 1)
    else:
        w_norm = weights / w_sum

    out: Dict[str, Any] = {
        "R_m": float(np.sum(w_norm * df["R_m"].astype(float).to_numpy())),
        "K_m": float(np.sum(w_norm * df["K_m"].astype(float).to_numpy())),
        "r_noise": float(np.sum(w_norm * df["r_noise"].astype(float).to_numpy())),
    }
    for _, row in df.iterrows():
        macro = str(row.get("macro", ""))
        if macro not in MACRO_NAMES:
            continue
        out[f"R_m_{macro}"] = float(row["R_m"])
        out[f"K_m_{macro}"] = float(row["K_m"])
        out[f"r_noise_{macro}"] = float(row["r_noise"])
    return out


def _metrics_phase_block(metrics: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {
        f"macro_f1_{prefix}": _safe_float(metrics.get("macro_f1")),
        f"balanced_accuracy_{prefix}": _safe_float(metrics.get("balanced_accuracy")),
        f"entropy_{prefix}": _safe_float(metrics.get("mean_entropy")),
        f"accuracy_{prefix}": _safe_float(metrics.get("accuracy")),
        f"mean_q_conf_{prefix}": _safe_float(metrics.get("mean_q_conf")),
        f"n_eval_{prefix}": metrics.get("n_eval"),
    }


def collect_run_metrics_row(
    *,
    corpus_id: str,
    base_method: str,
    method_name: str,
    lambda_pres: float,
    output_dir: Path,
    checkpoint: str,
) -> Dict[str, Any]:
    """Une ligne de métriques pour un run TPN (initial + adapté + topics)."""
    out = Path(output_dir)
    m_init, m_adapt = load_transfer_metrics_pair(out)

    row: Dict[str, Any] = {
        "corpus": corpus_id,
        "base_method": base_method,
        "method": method_name,
        "lambda_pres": float(lambda_pres),
        "checkpoint": str(checkpoint),
        "output_dir": str(out),
    }
    row.update(_metrics_phase_block(m_init, "initial"))
    row.update(_metrics_phase_block(m_adapt, "adapted"))

    stats_path = out / "summary" / "macro_topic_stats.csv"
    if stats_path.is_file():
        stats_df = pd.read_csv(stats_path)
        row.update(aggregate_topic_metrics_from_stats(stats_df))
    else:
        row.update({"R_m": float("nan"), "K_m": float("nan"), "r_noise": float("nan")})

    manifest_path = out / "run_manifest.json"
    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        row["encode_skipped"] = manifest.get("encode_skipped")
        row["projected_cache_dir"] = manifest.get("projected_cache_dir")

    return row


def metrics_rows_to_dataframe(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "corpus",
                "base_method",
                "lambda_pres",
                "macro_f1_initial",
                "macro_f1_adapted",
                "balanced_accuracy_initial",
                "balanced_accuracy_adapted",
                "entropy_initial",
                "entropy_adapted",
                "R_m",
                "K_m",
                "r_noise",
            ]
        )
    df = pd.DataFrame(rows)
    if "lambda_pres" in df.columns:
        df = df.sort_values(["base_method", "lambda_pres"]).reset_index(drop=True)
    return df


def write_preserve_tuning_csv(
    rows: Sequence[Dict[str, Any]],
    path: Path,
) -> pd.DataFrame:
    df = metrics_rows_to_dataframe(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def parse_lambda_pres_list(raw: str) -> List[float]:
    parts = [p.strip() for p in re.split(r"[,;\s]+", raw.strip()) if p.strip()]
    return [float(p) for p in parts]


def parse_base_methods_list(raw: Optional[str]) -> List[str]:
    if not raw or not str(raw).strip():
        return list(ALL_TPN_ENCODERS)
    return [m.strip() for m in str(raw).split(",") if m.strip()]
