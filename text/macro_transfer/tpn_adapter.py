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
    compute_source_prototypes,
    compute_source_target_prototypes,
    compute_target_prototypes_soft,
    distribution_from_prototypes_torch,
    l2_normalize_np,
    scores_from_prototypes,
    soft_assignments,
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


def _entropy_rows(p: torch.Tensor) -> float:
    p = torch.clamp(p, min=EPS)
    return float((-p * torch.log(p)).sum(dim=1).mean().item())


def _compute_tpn_losses(
    adapter: nn.Module,
    h_s: torch.Tensor,
    h_t: torch.Tensor,
    y_ids: torch.Tensor,
    *,
    tpn_cfg: Dict[str, Any],
    loss_weights: Dict[str, float],
) -> Dict[str, Any]:
    tau = float(tpn_cfg.get("tau", 0.3))
    metric = str(tpn_cfg.get("distance_metric", "euclidean"))
    rho = float(tpn_cfg.get("target_weight_st", 1.0))
    detach_q = bool(tpn_cfg.get("detach_assignments", True))
    n_macros = len(MACRO_NAMES)

    htilde_s = adapter(h_s)
    htilde_t = adapter(h_t)

    labels_s = [MACRO_NAMES[int(i)] for i in y_ids.cpu().numpy()]
    mu_s_np = compute_source_prototypes(
        htilde_s.detach().cpu().numpy(), labels_s, macros=MACRO_NAMES
    )
    mu_s = torch.as_tensor(mu_s_np, dtype=htilde_s.dtype, device=htilde_s.device)

    h_for_q = htilde_t.detach().cpu().numpy() if detach_q else htilde_t.cpu().numpy()
    scores_t = scores_from_prototypes(h_for_q, mu_s_np, tau=tau, metric=metric)  # type: ignore[arg-type]
    q_np = soft_assignments(
        scores_t,
        assignment_mode=str(tpn_cfg.get("assignment_mode", "soft")),  # type: ignore[arg-type]
    )
    q = torch.as_tensor(q_np, dtype=htilde_s.dtype, device=htilde_s.device)
    if detach_q:
        q = q.detach()

    mu_t_np = compute_target_prototypes_soft(htilde_t.detach().cpu().numpy(), q_np, eps=EPS)
    mu_t = torch.as_tensor(mu_t_np, dtype=htilde_s.dtype, device=htilde_s.device)

    mu_st_np = compute_source_target_prototypes(
        htilde_s.detach().cpu().numpy(),
        labels_s,
        htilde_t.detach().cpu().numpy(),
        q_np,
        rho=rho,
        eps=EPS,
        macros=MACRO_NAMES,
    )
    mu_st = torch.as_tensor(mu_st_np, dtype=htilde_s.dtype, device=htilde_s.device)

    p_st_src = distribution_from_prototypes_torch(
        htilde_s, mu_st, tau=tau, metric=metric  # type: ignore[arg-type]
    )
    loss_src = F.nll_loss(torch.log(p_st_src + EPS), y_ids)

    loss_proto = (
        ((mu_s - mu_t) ** 2).sum()
        + ((mu_s - mu_st) ** 2).sum()
        + ((mu_t - mu_st) ** 2).sum()
    ) / (3.0 * n_macros)

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

    def _skl_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            (a * (torch.log(a + EPS) - torch.log(b + EPS))).sum(dim=1)
            + (b * (torch.log(b + EPS) - torch.log(a + EPS))).sum(dim=1)
        )

    loss_kl = (_skl_batch(p_s_all, p_t_all) + _skl_batch(p_s_all, p_st_all) + _skl_batch(p_t_all, p_st_all)).mean()

    p_st_tgt = distribution_from_prototypes_torch(
        htilde_t, mu_st, tau=tau, metric=metric  # type: ignore[arg-type]
    )
    loss_ent = _entropy_rows(p_st_tgt)

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
        "loss_ent": loss_ent,
        "loss_div": loss_div.detach(),
        "loss_pres": loss_pres.detach(),
    }


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
    early_stopping_patience: int = 10,
    min_delta: float = 1e-5,
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

    log_rows: list[dict] = []
    best_loss = float("inf")
    best_state: Optional[dict] = None
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        adapter.train()
        opt.zero_grad()
        losses = _compute_tpn_losses(
            adapter, h_s_t, h_t_t, y_ids, tpn_cfg=tpn_cfg, loss_weights=loss_weights
        )
        losses["loss_total"].backward()
        opt.step()

        row: dict = {"epoch": epoch}
        for k, v in losses.items():
            row[k] = float(v.item() if hasattr(v, "item") else v)
        log_rows.append(row)

        if row["loss_total"] < best_loss - min_delta:
            best_loss = row["loss_total"]
            best_state = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= early_stopping_patience:
                logger.info("Early stopping epoch %d (best_loss=%.6f)", epoch, best_loss)
                break

    if best_state is not None:
        adapter.load_state_dict(best_state)

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(log_rows).to_csv(log_path, index=False)

    return adapter, pd.DataFrame(log_rows)
