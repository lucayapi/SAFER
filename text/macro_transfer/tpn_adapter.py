"""Adaptateur TPN (g_phi) et boucle d'entraînement."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from macro_transfer.constants import LABEL2ID, MACRO_NAMES
from macro_transfer.tpn_prototypes import (
    compute_source_prototypes_torch,
    compute_source_target_prototypes_torch,
    compute_target_prototypes_soft_torch,
    distribution_from_prototypes_torch,
    l2_normalize_np,
    prototype_distance_torch,
    prototype_logits_torch,
    soft_assignments_torch,
    symmetric_kl_torch,
)

logger = logging.getLogger(__name__)

EPS = 1e-8


class ResidualMLPAdapter(nn.Module):
    """g(h) = normalize(h + scale * MLP(LN(h)))."""

    def __init__(
        self,
        dim: int,
        *,
        bottleneck_dim: int = 256,
        dropout: float = 0.05,
        scale: float = 0.1,
        init_last_zero: bool = True,
    ):
        super().__init__()
        self.scale = float(scale)
        self.ln = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, dim),
        )
        if init_last_zero:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        delta = self.mlp(self.ln(h))
        out = h + self.scale * delta
        return F.normalize(out, p=2, dim=-1, eps=EPS)


class LinearAdapter(nn.Module):
    """g(h) = normalize(Ah + b), A proche identité."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=True)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.linear(h), p=2, dim=-1, eps=EPS)


def build_adapter(cfg: Dict[str, Any], dim: int) -> nn.Module:
    adapter_type = str(cfg.get("type", "residual_mlp"))
    if adapter_type == "linear":
        return LinearAdapter(dim)
    return ResidualMLPAdapter(
        dim,
        bottleneck_dim=int(cfg.get("bottleneck_dim", 256)),
        dropout=float(cfg.get("dropout", 0.05)),
        scale=float(cfg.get("scale", 0.1)),
        init_last_zero=bool(cfg.get("init_last_zero", True)),
    )


def adapt_embeddings_tpn(adapter: nn.Module, h: np.ndarray, *, device: str = "cpu") -> np.ndarray:
    adapter.eval()
    with torch.no_grad():
        t = torch.as_tensor(h, dtype=torch.float32, device=device)
        out = adapter(t)
        return out.cpu().numpy().astype(np.float64)


def _compute_tpn_losses(
    adapter: nn.Module,
    h_s: torch.Tensor,
    h_t: torch.Tensor,
    y_ids: torch.Tensor,
    *,
    tpn_cfg: Dict[str, Any],
    loss_weights: Dict[str, float],
) -> Dict[str, Any]:
    """Pertes TPN entièrement en Torch (graphe vers l'adaptateur)."""
    tau = float(tpn_cfg.get("tau", 0.3))
    metric = str(tpn_cfg.get("distance_metric", "euclidean"))  # type: ignore[assignment]
    rho = float(tpn_cfg.get("target_weight_st", 1.0))
    detach_q = bool(tpn_cfg.get("detach_assignments", True))
    assignment_mode = str(tpn_cfg.get("assignment_mode", "soft"))  # type: ignore[assignment]
    n_macros = len(MACRO_NAMES)

    htilde_s = adapter(h_s)
    htilde_t = adapter(h_t)

    mu_s = compute_source_prototypes_torch(htilde_s, y_ids, n_macros, eps=EPS)

    logits_q = prototype_logits_torch(htilde_t, mu_s, tau=tau, metric=metric)  # type: ignore[arg-type]
    q = soft_assignments_torch(logits_q, assignment_mode=assignment_mode)  # type: ignore[arg-type]
    if detach_q:
        q = q.detach()

    mu_t = compute_target_prototypes_soft_torch(htilde_t, q, eps=EPS)
    mu_st = compute_source_target_prototypes_torch(
        htilde_s, y_ids, htilde_t, q, n_macros, rho=rho, eps=EPS
    )

    logits_src = prototype_logits_torch(htilde_s, mu_st, tau=tau, metric=metric)  # type: ignore[arg-type]
    loss_src = F.cross_entropy(logits_src, y_ids)

    proto_terms = []
    for m in range(n_macros):
        proto_terms.append(prototype_distance_torch(mu_s[m], mu_t[m], metric=metric))  # type: ignore[arg-type]
        proto_terms.append(prototype_distance_torch(mu_s[m], mu_st[m], metric=metric))  # type: ignore[arg-type]
        proto_terms.append(prototype_distance_torch(mu_t[m], mu_st[m], metric=metric))  # type: ignore[arg-type]
    loss_proto = torch.stack(proto_terms).mean()

    combined = torch.cat([htilde_s, htilde_t], dim=0)
    p_s_all = distribution_from_prototypes_torch(
        combined, mu_s, tau=tau, metric=metric  # type: ignore[arg-type]
    )
    p_t_all = distribution_from_prototypes_torch(
        combined, mu_t, tau=tau, metric=metric  # type: ignore[arg-type]
    )
    p_st_all = distribution_from_prototypes_torch(
        combined, mu_st, tau=tau, metric=metric  # type: ignore[arg-type]
    )
    loss_kl = (
        symmetric_kl_torch(p_s_all, p_t_all, eps=EPS)
        + symmetric_kl_torch(p_s_all, p_st_all, eps=EPS)
        + symmetric_kl_torch(p_t_all, p_st_all, eps=EPS)
    ).mean()

    p_st_tgt = distribution_from_prototypes_torch(
        htilde_t, mu_st, tau=tau, metric=metric  # type: ignore[arg-type]
    )
    p_clamped = p_st_tgt.clamp(min=EPS)
    loss_ent = -(p_clamped * torch.log(p_clamped)).sum(dim=1).mean()

    p_bar = p_st_tgt.mean(dim=0)
    loss_div = (p_bar * torch.log(p_bar + EPS)).sum()

    loss_pres = ((htilde_s - h_s) ** 2).sum(dim=1).mean() + ((htilde_t - h_t) ** 2).sum(dim=1).mean()

    w = loss_weights
    total = (
        float(w.get("src", 1.0)) * loss_src
        + float(w.get("proto", 1.0)) * loss_proto
        + float(w.get("kl", 0.5)) * loss_kl
        + float(w.get("ent", 0.05)) * loss_ent
        + float(w.get("div", 0.05)) * loss_div
        + float(w.get("preserve", 0.1)) * loss_pres
    )

    return {
        "loss_total": total,
        "loss_src": loss_src.detach(),
        "loss_proto": loss_proto.detach(),
        "loss_kl": loss_kl.detach(),
        "loss_ent": loss_ent.detach(),
        "loss_div": loss_div.detach(),
        "loss_pres": loss_pres.detach(),
    }


def _losses_row_to_dict(losses: Dict[str, Any]) -> Dict[str, float]:
    row: Dict[str, float] = {}
    for k, v in losses.items():
        row[k] = float(v.item() if hasattr(v, "item") else v)
    return row


def _flush_training_log(log_path: Optional[Path], log_rows: list[dict]) -> None:
    """Réécrit le CSV à chaque epoch (suivi via ``tail -f`` pendant Slurm)."""
    if log_path is None or not log_rows:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(path, index=False)


def _log_epoch_progress(epoch: int, epochs: int, row: dict) -> None:
    logger.info(
        "TPN train epoch %d/%d | loss_total=%.6f | "
        "src=%.6f proto=%.6f kl=%.6f ent=%.6f div=%.6f pres=%.6f",
        epoch,
        epochs,
        row["loss_total"],
        row.get("loss_src", 0.0),
        row.get("loss_proto", 0.0),
        row.get("loss_kl", 0.0),
        row.get("loss_ent", 0.0),
        row.get("loss_div", 0.0),
        row.get("loss_pres", 0.0),
    )


def train_tpn_adapter(
    h_s: np.ndarray,
    h_t: np.ndarray,
    labels_s: np.ndarray,
    *,
    adapter_cfg: Dict[str, Any],
    tpn_cfg: Dict[str, Any],
    loss_weights: Dict[str, Any],
    epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 42,
    log_path: Optional[Path] = None,
) -> tuple[nn.Module, pd.DataFrame]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    dim = h_s.shape[1]
    adapter = build_adapter(adapter_cfg, dim).to(device)
    h_s_t = torch.as_tensor(l2_normalize_np(h_s), dtype=torch.float32, device=device)
    h_t_t = torch.as_tensor(l2_normalize_np(h_t), dtype=torch.float32, device=device)
    y_ids = torch.as_tensor(
        [LABEL2ID[str(y)] for y in labels_s],
        dtype=torch.long,
        device=device,
    )

    opt = torch.optim.AdamW(adapter.parameters(), lr=learning_rate, weight_decay=weight_decay)

    log_path_resolved = Path(log_path) if log_path is not None else None
    if log_path_resolved is not None:
        log_path_resolved.parent.mkdir(parents=True, exist_ok=True)
        logger.info("TPN adapter training: %d epochs → %s", epochs, log_path_resolved)
    else:
        logger.info("TPN adapter training: %d epochs (pas de training_log.csv)", epochs)

    log_rows: list[dict] = []

    for epoch in range(1, epochs + 1):
        adapter.train()
        opt.zero_grad()
        losses = _compute_tpn_losses(
            adapter, h_s_t, h_t_t, y_ids, tpn_cfg=tpn_cfg, loss_weights=loss_weights
        )
        losses["loss_total"].backward()
        opt.step()

        if epoch == 1:
            grad_norm = 0.0
            for p in adapter.parameters():
                if p.grad is not None:
                    grad_norm += float(p.grad.data.norm(2).item()) ** 2
            grad_norm = grad_norm**0.5
            logger.info(
                "TPN epoch 1 diag: grad_norm=%.4f loss_proto.requires_grad=%s loss_ent.requires_grad=%s",
                grad_norm,
                losses["loss_proto"].requires_grad,
                losses["loss_ent"].requires_grad,
            )

        row = {"epoch": epoch, **_losses_row_to_dict(losses)}
        log_rows.append(row)

        _log_epoch_progress(epoch, epochs, row)
        _flush_training_log(log_path_resolved, log_rows)

    _flush_training_log(log_path_resolved, log_rows)
    if log_path_resolved is not None and log_rows:
        loss_final = log_rows[-1]["loss_total"]
        logger.info(
            "TPN entraînement terminé : %d epochs enregistrées, loss_final=%.6f → %s",
            len(log_rows),
            loss_final,
            log_path_resolved,
        )

    return adapter, pd.DataFrame(log_rows)
