"""Évaluation transfert TPN (labels cible uniquement en post-hoc)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from macro_transfer.constants import LABEL2ID, MACRO_NAMES, VALID_LABELS
from scgm_text.metrics import accuracy, balanced_accuracy, macro_f1

CONFIDENCE_THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
MARGIN_THRESHOLDS = [0.00, 0.02, 0.03, 0.05, 0.08, 0.10]


def _valid_label_mask(series: pd.Series) -> np.ndarray:
    return series.notna() & series.astype(str).isin(VALID_LABELS)


def evaluate_tpn_transfer(
    meta: pd.DataFrame,
    *,
    label_col: str = "pred_label",
    pred_ok_col: Optional[str] = "pred_ok",
    m_hat_col: str = "m_hat",
) -> Dict[str, Any]:
    from scgm_text.dataset_text_embeddings import parse_bool_column

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

    metrics: Dict[str, Any] = {
        "n_eval": int(len(y_true_id)),
        "accuracy": accuracy(y_true_id, y_pred_id),
        "macro_f1": macro_f1(y_true_id, y_pred_id),
        "balanced_accuracy": balanced_accuracy(y_true_id, y_pred_id),
        "confusion": {
            MACRO_NAMES[i]: {MACRO_NAMES[j]: int(cm[i, j]) for j in range(4)} for i in range(4)
        },
        "mean_q_conf": float(sub.loc[mask, "q_conf"].mean()) if "q_conf" in sub.columns else float("nan"),
        "mean_margin": float(sub.loc[mask, "margin"].mean()) if "margin" in sub.columns else float("nan"),
        "mean_entropy": float(sub.loc[mask, "entropy"].mean()) if "entropy" in sub.columns else float("nan"),
    }

    try:
        from sklearn.metrics import classification_report

        metrics["classification_report"] = classification_report(
            y_true, y_pred, labels=list(MACRO_NAMES), output_dict=True, zero_division=0
        )
    except ImportError:
        metrics["classification_report"] = None

    return metrics


def save_tpn_eval(metrics: Dict[str, Any], output_dir: Path, prefix: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in metrics.items() if k != "classification_report"}
    with open(output_dir / f"transfer_metrics_{prefix}.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    rows = []
    conf = metrics.get("confusion") or {}
    for true_m in MACRO_NAMES:
        row = {"true": true_m}
        row.update({f"pred_{p}": conf.get(true_m, {}).get(p, 0) for p in MACRO_NAMES})
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / f"transfer_metrics_{prefix}.csv", index=False)


def compute_coverage_by_threshold(
    meta: pd.DataFrame,
    *,
    label_col: str = "pred_label",
    pred_ok_col: Optional[str] = "pred_ok",
    m_hat_col: str = "m_hat",
    confidence_thresholds: Optional[List[float]] = None,
    margin_thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    from scgm_text.dataset_text_embeddings import parse_bool_column

    confidence_thresholds = confidence_thresholds or CONFIDENCE_THRESHOLDS
    margin_thresholds = margin_thresholds or MARGIN_THRESHOLDS

    mask = _valid_label_mask(meta[label_col])
    if pred_ok_col and pred_ok_col in meta.columns:
        mask = mask & parse_bool_column(meta[pred_ok_col])
    if not mask.any():
        return pd.DataFrame()

    sub = meta.loc[mask].copy()
    y_true = sub[label_col].astype(str).to_numpy()
    y_pred = sub[m_hat_col].astype(str).to_numpy()
    y_true_id = np.array([LABEL2ID.get(y, -1) for y in y_true], dtype=np.int64)
    y_pred_id = np.array([LABEL2ID.get(y, -1) for y in y_pred], dtype=np.int64)

    rows: list[dict] = []
    n_total = len(sub)

    for c_thr in confidence_thresholds:
        keep = sub["q_conf"].astype(float) >= c_thr
        n_kept = int(keep.sum())
        if n_kept == 0:
            rows.append(
                {
                    "threshold_type": "confidence",
                    "threshold": c_thr,
                    "coverage": 0.0,
                    "n_kept": 0,
                    "accuracy": float("nan"),
                    "macro_f1": float("nan"),
                    "balanced_accuracy": float("nan"),
                }
            )
            continue
        yt = y_true_id[keep.to_numpy()]
        yp = y_pred_id[keep.to_numpy()]
        ok = (yt >= 0) & (yp >= 0)
        rows.append(
            {
                "threshold_type": "confidence",
                "threshold": c_thr,
                "coverage": n_kept / n_total,
                "n_kept": n_kept,
                "accuracy": accuracy(yt[ok], yp[ok]) if ok.any() else float("nan"),
                "macro_f1": macro_f1(yt[ok], yp[ok]) if ok.any() else float("nan"),
                "balanced_accuracy": balanced_accuracy(yt[ok], yp[ok]) if ok.any() else float("nan"),
            }
        )

    for m_thr in margin_thresholds:
        keep = sub["margin"].astype(float) >= m_thr
        n_kept = int(keep.sum())
        if n_kept == 0:
            rows.append(
                {
                    "threshold_type": "margin",
                    "threshold": m_thr,
                    "coverage": 0.0,
                    "n_kept": 0,
                    "accuracy": float("nan"),
                    "macro_f1": float("nan"),
                    "balanced_accuracy": float("nan"),
                }
            )
            continue
        yt = y_true_id[keep.to_numpy()]
        yp = y_pred_id[keep.to_numpy()]
        ok = (yt >= 0) & (yp >= 0)
        rows.append(
            {
                "threshold_type": "margin",
                "threshold": m_thr,
                "coverage": n_kept / n_total,
                "n_kept": n_kept,
                "accuracy": accuracy(yt[ok], yp[ok]) if ok.any() else float("nan"),
                "macro_f1": macro_f1(yt[ok], yp[ok]) if ok.any() else float("nan"),
                "balanced_accuracy": balanced_accuracy(yt[ok], yp[ok]) if ok.any() else float("nan"),
            }
        )

    return pd.DataFrame(rows)
