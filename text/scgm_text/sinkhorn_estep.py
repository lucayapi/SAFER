"""Sinkhorn-Knopp E-step with configurable sample marginals."""

from __future__ import annotations

import json
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from sinkhornknopp import optimize_l_sk

SINKHORN_SAMPLE_PRIORS = frozenset({"uniform", "macro_balanced"})

# In the original SCGM paper, b_i = 1/n gives each sample equal mass.
# For imbalanced macro labels, macro_balanced uses b_i = 1/(c_present * n_{y_i})
# so that each macro contributes the same total mass to the Sinkhorn E-step.


def _normalize_sample_prior(sample_prior: str) -> str:
    mode = str(sample_prior).strip().lower()
    if mode not in SINKHORN_SAMPLE_PRIORS:
        raise ValueError(
            f"sinkhorn_sample_prior must be one of: {', '.join(sorted(SINKHORN_SAMPLE_PRIORS))} "
            f"(got {sample_prior!r})"
        )
    return mode


def _labels_to_long_tensor(
    labels: Union[torch.Tensor, np.ndarray],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if isinstance(labels, np.ndarray):
        lab = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    elif torch.is_tensor(labels):
        lab = labels.detach().cpu()
    else:
        raise TypeError(f"labels must be torch.Tensor or np.ndarray, got {type(labels)!r}")

    if lab.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {tuple(lab.shape)}")
    return lab.to(dtype=torch.long, device=device or torch.device("cpu"))


def build_sinkhorn_marginals(
    labels: Union[torch.Tensor, np.ndarray],
    n_latents: int,
    n_classes: int,
    sample_prior: str = "uniform",
    device: Optional[torch.device] = None,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build Sinkhorn marginals a (latent) and b (sample).

    a_z = 1/r always. b_i is 1/n (uniform) or 1/(c_present * n_{y_i}) (macro_balanced).
    """
    mode = _normalize_sample_prior(sample_prior)
    dev = device or torch.device("cpu")
    lab = _labels_to_long_tensor(labels, device=dev)
    n = int(lab.numel())
    r = int(n_latents)
    if n == 0:
        raise ValueError("labels must be non-empty for Sinkhorn marginals.")
    if r <= 0:
        raise ValueError("n_latents must be positive.")

    if lab.min() < 0 or lab.max() >= n_classes:
        raise ValueError(
            f"label ids must be in [0, {n_classes}), got min={int(lab.min())} max={int(lab.max())}"
        )

    a = torch.ones(r, device=dev, dtype=torch.float64) / float(r)

    if mode == "uniform":
        b = torch.ones(n, device=dev, dtype=torch.float64) / float(n)
    else:
        counts = torch.bincount(lab, minlength=n_classes).to(dtype=torch.float64)
        present_mask = counts > 0
        c_present = int(present_mask.sum().item())
        if c_present == 0:
            raise ValueError("No macro class present in labels for macro_balanced prior.")
        denom = float(c_present) * counts[lab].clamp(min=eps)
        b = (1.0 / denom).to(dtype=torch.float64)

    _assert_marginals(a, b, atol=1e-5)
    return a, b


def macro_masses_in_b(
    b: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    n_classes: int = 4,
) -> Dict[int, float]:
    """Total mass sum(b_i) for each macro m: mass_m = b[labels == m].sum()."""
    b_t = b if torch.is_tensor(b) else torch.tensor(b, dtype=torch.float64)
    lab = _labels_to_long_tensor(labels, device=b_t.device)
    masses: Dict[int, float] = {}
    for m in range(n_classes):
        mask = lab == m
        if mask.any():
            masses[m] = float(b_t[mask].sum().item())
    return masses


def _assert_marginals(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-5) -> None:
    assert a.ndim == 1 and b.ndim == 1
    assert torch.all(a >= 0) and torch.all(b >= 0)
    assert torch.isclose(a.sum(), torch.tensor(1.0, device=a.device, dtype=a.dtype), atol=atol)
    assert torch.isclose(b.sum(), torch.tensor(1.0, device=b.device, dtype=b.dtype), atol=atol)


def _log_sinkhorn_marginals(
    sample_prior: str,
    a: torch.Tensor,
    b: torch.Tensor,
    labels: Union[torch.Tensor, np.ndarray],
    n_classes: int,
) -> None:
    masses = macro_masses_in_b(b, labels, n_classes=n_classes)
    mass_json = json.dumps({str(k): round(v, 4) for k, v in sorted(masses.items())})
    print(f"[SCGM Sinkhorn] sample_prior={sample_prior}", flush=True)
    print(
        f"[SCGM Sinkhorn] a_sum={float(a.sum()):.4f} b_sum={float(b.sum()):.4f}",
        flush=True,
    )
    print(f"[SCGM Sinkhorn] macro mass in b: {mass_json}", flush=True)


def sinkhorn_assign(
    score_matrix: np.ndarray,
    lmd: float,
    *,
    labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
    n_classes: int = 4,
    sample_prior: str = "uniform",
    log_marginals: bool = False,
    max_iter: int = 500,
    tol: float = 1e-4,
    tol_mode: str = "mean",
    check_every: int = 10,
    eps: float = 1e-12,
    normalize_input: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Assign latent components via Sinkhorn-Knopp.

    Parameters
    ----------
    score_matrix : (n, r) scores P_sample,z; internally transposed to (r, n) for OT.
    labels : (n,) macro class ids aligned with rows of score_matrix.
    """
    scores = np.asarray(score_matrix, dtype=np.float64)
    n_samples, n_latents = scores.shape

    a_np: Optional[np.ndarray] = None
    b_np: Optional[np.ndarray] = None

    if labels is not None:
        lab_t = _labels_to_long_tensor(labels)
        if lab_t.numel() != n_samples:
            raise ValueError(
                f"labels length {lab_t.numel()} != score_matrix rows {n_samples}"
            )
        a_t, b_t = build_sinkhorn_marginals(
            lab_t,
            n_latents=n_latents,
            n_classes=n_classes,
            sample_prior=sample_prior,
        )
        assert scores.shape[0] == b_t.shape[0]
        assert scores.shape[1] == a_t.shape[0]
        _assert_marginals(a_t, b_t)
        if log_marginals:
            _log_sinkhorn_marginals(sample_prior, a_t, b_t, lab_t, n_classes)
        a_np = a_t.detach().cpu().numpy()
        b_np = b_t.detach().cpu().numpy()

    prob, argmax_q, sk_meta = optimize_l_sk(
        scores,
        lmd,
        a=a_np,
        b=b_np,
        max_iter=max_iter,
        tol=tol,
        tol_mode=tol_mode,
        check_every=check_every,
        eps=eps,
        normalize_input=normalize_input,
        verbose=verbose,
    )
    prob = np.asarray(prob, dtype=np.float64)
    row_sums = prob.sum(axis=1, keepdims=True)
    row_sums = np.clip(row_sums, eps, None)
    prob_norm = prob / row_sums
    entropy = -np.sum(prob_norm * np.log(np.clip(prob_norm, eps, None)))
    active = int(np.unique(argmax_q).size)
    diagnostics = {
        "sinkhorn_assignment_entropy": float(entropy / max(prob.shape[0], 1)),
        "sinkhorn_n_active_z": float(active),
        "sinkhorn_mean_row_mass": float(prob.sum(axis=1).mean()),
        "sinkhorn_sample_prior": sample_prior,
        "sinkhorn_converged": float(sk_meta.get("converged", False)),
        "sinkhorn_n_iter": float(sk_meta.get("n_iter", 0)),
        "sinkhorn_err_final": float(sk_meta.get("err_final", float("nan"))),
        "sinkhorn_err_sum_final": float(sk_meta.get("err_sum_final", float("nan"))),
        "sinkhorn_err_mean_final": float(sk_meta.get("err_mean_final", float("nan"))),
        "sinkhorn_tol_mode": str(sk_meta.get("tol_mode", tol_mode)),
        "sinkhorn_tol": float(sk_meta.get("tol", tol)),
        "sinkhorn_max_iter": float(sk_meta.get("max_iter", max_iter)),
        "sinkhorn_lmd_effective": float(sk_meta.get("lmd", lmd)),
    }
    return prob, argmax_q, diagnostics
