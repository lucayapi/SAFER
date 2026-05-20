"""Probabilités macro p(m|u) depuis centres SoftTriple."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from contrastive_methods.distance import embedding_to_center_scores
from macro_transfer.constants import MACRO_NAMES


def _l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return x / norms


def load_softtriple_centers(
    checkpoint_dir: Path,
    *,
    centers_json: Optional[Path] = None,
) -> np.ndarray:
    """
    Charge centres [C, K, D]. Priorité : JSON centres effectifs, sinon softtriple_state.pt.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.is_file():
        checkpoint_dir = checkpoint_dir.parent

    if centers_json and Path(centers_json).is_file():
        with open(centers_json, encoding="utf-8") as f:
            data = json.load(f)
        arr = np.asarray(data.get("centers") or data.get("effective_centers"), dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[:, np.newaxis, :]
        return arr

    for name in ("effective_centers.json", "centers_effective.json"):
        p = checkpoint_dir / name
        if p.is_file():
            return load_softtriple_centers(checkpoint_dir, centers_json=p)

    state_path = checkpoint_dir / "softtriple_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"softtriple_state.pt introuvable dans {checkpoint_dir}")
    try:
        ckpt = torch.load(state_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(state_path, map_location="cpu")
    loss_state = ckpt.get("loss_state", ckpt)
    centers = loss_state["centers"]
    if hasattr(centers, "detach"):
        centers = centers.detach().cpu().numpy()
    return np.asarray(centers, dtype=np.float64)


def macro_probs_softtriple(
    z: np.ndarray,
    centers: np.ndarray,
    *,
    gamma: float = 0.1,
    temperature: float = 1.0,
    distance_metric: str = "euclidean",
    normalize_embeddings: bool = True,
    normalize_centers: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule p(m|u) : relaxed class similarity puis softmax / T.
    Retourne (prob_y, gamma_jmk) avec gamma_jmk = p(k|m,u) agrégé (N, C, K).
    """
    z_t = torch.from_numpy(np.asarray(z, dtype=np.float32))
    c_t = torch.from_numpy(np.asarray(centers, dtype=np.float32))
    if normalize_embeddings:
        z_t = F.normalize(z_t, p=2, dim=1)
    if normalize_centers:
        c_t = F.normalize(c_t, p=2, dim=-1)

    raw_sim = embedding_to_center_scores(z_t, c_t, metric=distance_metric)
    n_classes = raw_sim.shape[1]
    n_centers = raw_sim.shape[2]

    if n_centers == 1:
        relaxed = raw_sim.squeeze(-1)
        gamma_jmk = torch.ones(z_t.shape[0], n_classes, 1, device=raw_sim.device)
    else:
        gamma_jmk = F.softmax(raw_sim / max(float(gamma), 1e-8), dim=2)
        relaxed = (gamma_jmk * raw_sim).sum(dim=2)

    prob_y = F.softmax(relaxed / max(float(temperature), 1e-8), dim=1)
    return prob_y.detach().cpu().numpy(), gamma_jmk.detach().cpu().numpy()


def macro_probs_from_checkpoint(
    z: np.ndarray,
    checkpoint_dir: Path,
    *,
    gamma: float = 0.1,
    temperature: float = 1.0,
    distance_metric: str = "euclidean",
) -> np.ndarray:
    centers = load_softtriple_centers(Path(checkpoint_dir))
    if centers.shape[0] != len(MACRO_NAMES):
        raise ValueError(f"Attendu {len(MACRO_NAMES)} classes, centres shape {centers.shape}")
    prob_y, _ = macro_probs_softtriple(
        z,
        centers,
        gamma=gamma,
        temperature=temperature,
        distance_metric=distance_metric,
    )
    return prob_y
