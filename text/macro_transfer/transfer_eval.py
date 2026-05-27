"""Évaluation du transfert macro (classification vs pred_label)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from macro_transfer.constants import LABEL2ID, MACRO_NAMES, VALID_LABELS
from scgm_text.metrics import accuracy, balanced_accuracy, macro_f1


def _valid_label_mask(series: pd.Series) -> np.ndarray:
    return series.notna() & series.astype(str).isin(VALID_LABELS)


def evaluate_transfer_classification(
    meta: pd.DataFrame,
    *,
    label_col: str = "pred_label",
    pred_ok_col: Optional[str] = "pred_ok",
    m_hat_col: str = "m_hat",
) -> Dict[str, Any]:
    """Métriques de classification macro + matrice de confusion."""
    from scgm_text.utils_io import parse_bool_column

    mask = _valid_label_mask(meta[label_col])
    if pred_ok_col and pred_ok_col in meta.columns:
        mask = mask & parse_bool_column(meta[pred_ok_col])

    if not mask.any():
        return {
            "n_eval": 0,
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "balanced_accuracy": float("nan"),
            "confusion": {},
            "note": "aucune ligne avec label valide",
        }

    sub = meta.loc[mask]
    y_true = sub[label_col].astype(str).to_numpy()
    y_pred = sub[m_hat_col].astype(str).to_numpy()
    y_true_id = np.array([LABEL2ID.get(y, -1) for y in y_true], dtype=np.int64)
    y_pred_id = np.array([LABEL2ID.get(y, -1) for y in y_pred], dtype=np.int64)
    ok = (y_true_id >= 0) & (y_pred_id >= 0)
    y_true_id = y_true_id[ok]
    y_pred_id = y_pred_id[ok]

    cm = np.zeros((len(MACRO_NAMES), len(MACRO_NAMES)), dtype=np.int64)
    for t, p in zip(y_true_id, y_pred_id):
        cm[int(t), int(p)] += 1

    return {
        "n_eval": int(len(y_true_id)),
        "accuracy": accuracy(y_true_id, y_pred_id),
        "macro_f1": macro_f1(y_true_id, y_pred_id),
        "balanced_accuracy": balanced_accuracy(y_true_id, y_pred_id),
        "confusion": {MACRO_NAMES[i]: {MACRO_NAMES[j]: int(cm[i, j]) for j in range(4)} for i in range(4)},
        "mean_q_conf": float(sub.loc[mask, "q_conf"].mean()) if "q_conf" in sub.columns else float("nan"),
    }


def save_transfer_eval(
    metrics: Dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "transfer_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    rows = []
    conf = metrics.get("confusion") or {}
    for true_m in MACRO_NAMES:
        row = {"true": true_m}
        row.update({f"pred_{p}": conf.get(true_m, {}).get(p, 0) for p in MACRO_NAMES})
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "classification_report.csv", index=False)
