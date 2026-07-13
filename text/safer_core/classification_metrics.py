"""Métriques classification macro (accuracy, F1, balanced accuracy)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from scgm_text.dataset_text_embeddings import LABEL2ID


def evaluate_macro_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    probs: np.ndarray,
    macros: Sequence[str],
) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=object).astype(str)
    y_pred_arr = np.asarray(y_pred, dtype=object).astype(str)
    y_true_id = np.array([LABEL2ID.get(v, -1) for v in y_true_arr], dtype=np.int64)
    y_pred_id = np.array([LABEL2ID.get(v, -1) for v in y_pred_arr], dtype=np.int64)
    mask = (y_true_id >= 0) & (y_pred_id >= 0)
    if not bool(mask.any()):
        return {
            "n_eval": 0,
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "balanced_accuracy": float("nan"),
            "mean_confidence": float("nan"),
            "mean_margin": float("nan"),
            "mean_entropy": float("nan"),
            "classification_report": {},
            "confusion_matrix": np.zeros((len(macros), len(macros)), dtype=np.int64),
        }

    yt = y_true_id[mask]
    yp = y_pred_id[mask]
    p = np.asarray(probs, dtype=np.float64)[mask]
    p_sorted = np.sort(p, axis=1)
    return {
        "n_eval": int(len(yt)),
        "accuracy": float((yt == yp).mean()),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "mean_confidence": float(p.max(axis=1).mean()),
        "mean_margin": float((p_sorted[:, -1] - p_sorted[:, -2]).mean()) if p.shape[1] >= 2 else 0.0,
        "mean_entropy": float((-(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1)).mean()),
        "classification_report": classification_report(
            yt,
            yp,
            labels=list(range(len(macros))),
            target_names=list(macros),
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(yt, yp, labels=list(range(len(macros)))),
    }


def build_gating_from_predictions(preds: pd.DataFrame, macros: Sequence[str]) -> pd.DataFrame:
    """Construit un DataFrame gating compatible BERTopic intra-macro."""
    out = pd.DataFrame(index=preds.index.copy())
    out["m_hat"] = preds["pred_macro"].astype(str)
    out["ambiguous"] = False
    out["q_conf"] = pd.to_numeric(preds.get("confidence"), errors="coerce")
    for m in macros:
        pcol = f"prob_{m}"
        pcol_legacy = f"p_{m}"
        if pcol in preds.columns:
            vals = pd.to_numeric(preds[pcol], errors="coerce")
            out[pcol] = vals
            out[pcol_legacy] = vals
        else:
            out[pcol] = 0.0
            out[pcol_legacy] = 0.0
    return out
