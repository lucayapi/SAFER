"""Full-dataset E-step for MALT-EM (Sinkhorn on latent transfer scores)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from malt_text.malt_em_model import MALTEMTargetModel
from scgm_text.sinkhorn_estep import sinkhorn_assign

EPS = 1e-8


def _to_index_array(batch_item: Any) -> np.ndarray:
    if isinstance(batch_item, dict):
        idx = batch_item["index"]
    else:
        _, idx = batch_item
    if torch.is_tensor(idx):
        return idx.detach().cpu().numpy()
    return np.asarray(idx)


def compute_soft_macro_compatibility(
    p0: torch.Tensor,
    prob_y_z: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """
    psi[i,k] = prod_m p_t(y=m|z=k)^{p0[i,m]}  (shape N x K).
    """
    log_pyz = torch.log(prob_y_z.clamp_min(eps))
    soft_score = p0 @ log_pyz.transpose(0, 1)
    return torch.exp(soft_score)


def compute_malt_em_scores(
    prob_z_x: torch.Tensor,
    prob_y_z: torch.Tensor,
    p0: torch.Tensor,
    eps: float = EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    E-step scores P[i,k] = p_t(z=k|x_i) * exp(sum_m p0[i,m] log p_t(y=m|z=k)).

    All tensors: prob_z_x (N,K), prob_y_z (K,C), p0 (N,C).
    Returns scores and log_scores, both (N,K).
    """
    log_pz = torch.log(prob_z_x.clamp_min(eps))
    log_pyz = torch.log(prob_y_z.clamp_min(eps))
    log_scores = log_pz + p0 @ log_pyz.transpose(0, 1)
    return torch.exp(log_scores), log_scores


def q_entropy_per_row(q: np.ndarray, eps: float = EPS) -> float:
    qn = q / np.clip(q.sum(axis=1, keepdims=True), eps, None)
    ent = -np.sum(qn * np.log(np.clip(qn, eps, None)), axis=1)
    return float(np.mean(ent))


def hard_q_from_argmax(z_hat: np.ndarray, n_subclass: int) -> np.ndarray:
    q = np.zeros((len(z_hat), n_subclass), dtype=np.float32)
    q[np.arange(len(z_hat)), z_hat.astype(np.int64)] = 1.0
    return q


@torch.no_grad()
def run_malt_em_estep(
    model: MALTEMTargetModel,
    data_loader: DataLoader,
    p0_all: np.ndarray,
    tau_z: float,
    tau_yz: float,
    sinkhorn_lmd: float,
    device: torch.device,
    q_mode: str = "hard",
    eps: float = EPS,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Full-dataset E-step.

    Parameters
    ----------
    p0_all : (N, C) transferred soft macro responsibilities.
    score_matrix fed to Sinkhorn : (N, K) from compute_malt_em_scores.

    Returns
    -------
    q_all : (N, K)
    z_hat : (N,)
    diagnostics : dict
    """
    model.eval()
    n_total = len(p0_all)
    k = model.num_subclasses
    prob_z_parts: List[np.ndarray] = []
    index_parts: List[np.ndarray] = []
    prob_y_z_ref: Optional[torch.Tensor] = None

    for batch in data_loader:
        if isinstance(batch, dict):
            embeddings = batch["embedding"].to(device)
            indices = _to_index_array(batch)
        else:
            embeddings, indices_t = batch
            embeddings = embeddings.to(device)
            indices = indices_t.detach().cpu().numpy()

        probs = model.compute_all_probs(embeddings, tau_z=tau_z, tau_yz=tau_yz)
        prob_y_z_ref = probs["prob_y_z"]
        prob_z_parts.append(probs["prob_z_x"].detach().cpu().numpy())
        index_parts.append(indices)

    prob_z_all = np.zeros((n_total, k), dtype=np.float64)
    order = np.concatenate(index_parts, axis=0)
    prob_z_all[order] = np.concatenate(prob_z_parts, axis=0)

    p0_t = torch.from_numpy(p0_all).to(device=device, dtype=torch.float32)
    prob_y_z_t = prob_y_z_ref if prob_y_z_ref is not None else model.macro_given_latent(tau_yz)
    prob_z_t = torch.from_numpy(prob_z_all).to(device=device, dtype=torch.float32)

    scores_t, log_scores_t = compute_malt_em_scores(prob_z_t, prob_y_z_t, p0_t, eps=eps)
    scores = scores_t.detach().cpu().numpy()

    q_soft, z_hat, sk_diag = sinkhorn_assign(scores, sinkhorn_lmd)
    q_soft = np.asarray(q_soft, dtype=np.float64)
    row_sums = np.clip(q_soft.sum(axis=1, keepdims=True), eps, None)
    q_soft = q_soft / row_sums

    mode = str(q_mode).strip().lower()
    if mode == "hard":
        z_hat = np.asarray(z_hat, dtype=np.int64)
        q_all = hard_q_from_argmax(z_hat, k)
    elif mode == "soft":
        q_all = q_soft.astype(np.float32)
        z_hat = q_all.argmax(axis=1).astype(np.int64)
    else:
        raise ValueError(f"Unknown q_mode: {q_mode!r} (expected hard or soft)")

    pz_ent = float(
        -np.mean(
            np.sum(
                prob_z_all * np.log(np.clip(prob_z_all, eps, None)),
                axis=1,
            )
        )
    )
    pyz = prob_y_z_t.detach().cpu().numpy()
    pyz_ent = float(
        -np.mean(np.sum(pyz * np.log(np.clip(pyz, eps, None)), axis=1))
    )

    counts = np.bincount(z_hat, minlength=k)
    cluster_probs = counts / max(counts.sum(), 1)

    diagnostics: Dict[str, float] = {
        **{k: float(v) for k, v in sk_diag.items()},
        "q_entropy_mean": q_entropy_per_row(q_all),
        "pz_entropy_mean": pz_ent,
        "pyz_entropy_mean": pyz_ent,
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "n_active_z": float(np.unique(z_hat).size),
        "z_max_mass": float(cluster_probs.max()) if cluster_probs.size else 0.0,
    }
    for idx, count in enumerate(counts):
        diagnostics[f"q_count_z{idx}"] = float(count)

    return q_all.astype(np.float32), z_hat, diagnostics
