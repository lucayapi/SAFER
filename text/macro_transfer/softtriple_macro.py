"""Probabilités macro p(m|u) depuis centres SoftTriple."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from contrastive_methods.distance import embedding_to_center_scores
from macro_transfer.constants import MACRO_NAMES


def _l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return x / norms


def _load_softtriple_state(checkpoint_dir: Path) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.is_file():
        checkpoint_dir = checkpoint_dir.parent
    state_path = checkpoint_dir / "softtriple_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"softtriple_state.pt introuvable dans {checkpoint_dir}")
    try:
        ckpt = torch.load(state_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(state_path, map_location="cpu")
    return dict(ckpt)


def load_softtriple_centers(
    checkpoint_dir: Path,
    *,
    centers_json: Optional[Path] = None,
    prefer_raw_centers: bool = False,
) -> np.ndarray:
    """
    Charge centres [C, K, D]. Priorité : JSON centres effectifs, sinon softtriple_state.pt.
    Si ``prefer_raw_centers=True``, ignore les JSON et lit ``loss_state["centers"]``.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.is_file():
        checkpoint_dir = checkpoint_dir.parent

    if not prefer_raw_centers:
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

    ckpt = _load_softtriple_state(checkpoint_dir)
    loss_state = ckpt.get("loss_state", ckpt)
    centers = loss_state["centers"]
    if hasattr(centers, "detach"):
        centers = centers.detach().cpu().numpy()
    return np.asarray(centers, dtype=np.float64)


def load_softtriple_hyperparams(
    checkpoint_dir: Path,
    *,
    n_macros: int = len(MACRO_NAMES),
) -> dict[str, Any]:
    """Lit gamma, distance_metric, centers_per_class depuis softtriple_state.pt."""
    ckpt = _load_softtriple_state(checkpoint_dir)
    loss_state = ckpt.get("loss_state", ckpt)
    centers = loss_state.get("centers")
    if centers is None:
        raise KeyError(f"loss_state['centers'] absent dans {checkpoint_dir}")
    if hasattr(centers, "shape"):
        c_shape = tuple(centers.shape)
    else:
        c_shape = tuple(np.asarray(centers).shape)
    if c_shape[0] != n_macros:
        raise ValueError(f"Attendu {n_macros} classes, centres shape {c_shape}")

    cfg = dict(ckpt.get("config") or {})
    centers_per_class = int(cfg.get("centers_per_class", c_shape[1] if len(c_shape) > 1 else 1))
    return {
        "gamma": float(cfg.get("gamma", 0.1)),
        "distance_metric": str(cfg.get("distance_metric", "euclidean")),
        "centers_per_class": centers_per_class,
        "centers_shape": list(c_shape),
    }


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


def assign_macros_from_softtriple_centers(
    target_embeddings: np.ndarray,
    centers: np.ndarray,
    macros: Sequence[str],
    *,
    gamma: float,
    temperature: float,
    distance_metric: str,
    normalize_embeddings: bool = True,
    normalize_centers: bool = True,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """
    Affectation macro via centres SoftTriple natifs.
    ``distances`` = -S (score agrégé négatif) pour compatibilité colonnes dist_* FSP.
    """
    probs, gamma_jmk = macro_probs_softtriple(
        target_embeddings,
        centers,
        gamma=gamma,
        temperature=temperature,
        distance_metric=distance_metric,
        normalize_embeddings=normalize_embeddings,
        normalize_centers=normalize_centers,
    )
    z_t = torch.from_numpy(np.asarray(target_embeddings, dtype=np.float32))
    c_t = torch.from_numpy(np.asarray(centers, dtype=np.float32))
    if normalize_embeddings:
        z_t = F.normalize(z_t, p=2, dim=1)
    if normalize_centers:
        c_t = F.normalize(c_t, p=2, dim=-1)
    raw_sim = embedding_to_center_scores(z_t, c_t, metric=distance_metric)
    if raw_sim.shape[2] == 1:
        relaxed_scores = raw_sim.squeeze(-1).detach().cpu().numpy()
    else:
        g_t = torch.from_numpy(np.asarray(gamma_jmk, dtype=np.float32))
        relaxed_scores = (g_t * raw_sim).sum(dim=2).detach().cpu().numpy()

    top = probs.argmax(axis=1)
    pred_macro = np.array([str(macros[i]) for i in top], dtype=object)
    confidence = probs.max(axis=1)
    sort_p = np.sort(probs, axis=1)
    margin = sort_p[:, -1] - sort_p[:, -2] if probs.shape[1] >= 2 else np.zeros(len(probs))
    entropy = -(probs * np.log(np.clip(probs, eps, None))).sum(axis=1)
    distances = -np.asarray(relaxed_scores, dtype=np.float64)

    return {
        "pred_macro": pred_macro,
        "probs": probs,
        "distances": distances,
        "confidence": confidence,
        "margin": margin,
        "entropy": entropy,
        "gamma_jmk": gamma_jmk,
        "relaxed_scores": relaxed_scores,
    }


def export_softtriple_source_centers(
    centers: np.ndarray,
    macros: Sequence[str],
    *,
    normalize: bool = True,
) -> pd.DataFrame:
    """Exporte K lignes par macro : macro, center_k, prototype_norm, dim_*."""
    arr = np.asarray(centers, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[:, np.newaxis, :]
    if arr.shape[0] != len(macros):
        raise ValueError(f"centers.shape[0]={arr.shape[0]} != len(macros)={len(macros)}")

    rows: list[dict[str, Any]] = []
    vecs: list[np.ndarray] = []
    for i, macro in enumerate(macros):
        for k in range(arr.shape[1]):
            w = arr[i, k].copy()
            if normalize:
                denom = np.linalg.norm(w)
                if denom > 1e-12:
                    w = w / denom
            rows.append(
                {
                    "macro": str(macro),
                    "center_k": int(k),
                    "n_source": int(arr.shape[1]),
                    "prototype_norm": float(np.linalg.norm(w)),
                }
            )
            vecs.append(w)
    df = pd.DataFrame(rows)
    dim_cols = pd.DataFrame(
        np.asarray(vecs, dtype=np.float64),
        columns=[f"dim_{j:04d}" for j in range(arr.shape[-1])],
    )
    return pd.concat([df.reset_index(drop=True), dim_cols.reset_index(drop=True)], axis=1)


def summarize_center_weights(
    gamma_jmk: np.ndarray,
    macros: Sequence[str],
) -> pd.DataFrame:
    """Poids moyens et max par (macro, center_k) sur le corpus cible."""
    g = np.asarray(gamma_jmk, dtype=np.float64)
    if g.ndim != 3 or g.shape[1] != len(macros):
        raise ValueError(f"gamma_jmk attendu (N, C, K), reçu {g.shape}")
    rows: list[dict[str, Any]] = []
    for i, macro in enumerate(macros):
        for k in range(g.shape[2]):
            w = g[:, i, k]
            rows.append(
                {
                    "macro": str(macro),
                    "center_k": int(k),
                    "mean_weight": float(w.mean()),
                    "max_weight": float(w.max()),
                    "std_weight": float(w.std()),
                }
            )
    return pd.DataFrame(rows)


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
