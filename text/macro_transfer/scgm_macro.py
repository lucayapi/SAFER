"""Probabilités macro p(m|u) via modèle SCGM source."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from macro_transfer.constants import MACRO_NAMES
from scgm_text.checkpoint_io import load_scgm_checkpoint


def scgm_macro_probs(
    checkpoint_path: str,
    z: np.ndarray,
    *,
    tau: Optional[float] = None,
    device: str = "cuda",
    batch_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retourne (prob_y, prob_z, prob_y_z) pour embeddings projetés z (N, d).
    """
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model, args, _ = load_scgm_checkpoint(checkpoint_path, map_location="cpu")
    model.to(dev)
    model.eval()
    tau_val = float(tau if tau is not None else args.get("tau", 0.1))

    prob_y_list: list[np.ndarray] = []
    prob_z_list: list[np.ndarray] = []
    prob_y_z_list: list[np.ndarray] = []

    z_arr = np.asarray(z, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, z_arr.shape[0], batch_size):
            end = min(start + batch_size, z_arr.shape[0])
            x = torch.from_numpy(z_arr[start:end]).to(dev)
            x = F.normalize(x, p=2, dim=1)
            prob_y, prob_z, prob_y_z = model.pred(x, tau_val)
            prob_y_list.append(prob_y.cpu().numpy())
            prob_z_list.append(prob_z.cpu().numpy())
            prob_y_z_list.append(prob_y_z.cpu().numpy())

    return (
        np.concatenate(prob_y_list, axis=0),
        np.concatenate(prob_z_list, axis=0),
        np.concatenate(prob_y_z_list, axis=0),
    )


def load_z_to_macro_map(themes_by_z_csv: Path) -> dict[int, str]:
    """Mapping z_id → macro dominante depuis themes_by_z.csv source."""
    df = pd.read_csv(themes_by_z_csv)
    if "z_id" not in df.columns or "dominant_macro" not in df.columns:
        raise ValueError(f"Colonnes z_id / dominant_macro requises dans {themes_by_z_csv}")
    out: dict[int, str] = {}
    for _, row in df.iterrows():
        z_id = int(row["z_id"])
        macro = str(row.get("dominant_macro", "")).strip()
        if macro in MACRO_NAMES:
            out[z_id] = macro
    return out


def responsibility_by_macro(
    prob_z: np.ndarray,
    prob_y_z: np.ndarray,
    z_to_macro: dict[int, str],
) -> np.ndarray:
    """
    Diagnostic : r_{j,m} = sum_{k: macro(k)=m} p(z=k|u) * p(m|z=k) (pas topic final).
    """
    n, n_z = prob_z.shape
    n_m = len(MACRO_NAMES)
    r = np.zeros((n, n_m), dtype=np.float64)
    macro_to_idx = {m: i for i, m in enumerate(MACRO_NAMES)}
    for z_id in range(n_z):
        macro = z_to_macro.get(z_id)
        if macro not in macro_to_idx:
            continue
        mi = macro_to_idx[macro]
        r[:, mi] += prob_z[:, z_id] * prob_y_z[z_id, mi]
    row_sum = r.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 1e-12, row_sum, 1.0)
    return r / row_sum
