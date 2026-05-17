"""M-step losses for MALT-EM (fixed q from E-step; no Sinkhorn here)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from malt_text.utils import EPS, safe_log


@dataclass
class MALTEMLossOutput:
    loss_total: torch.Tensor
    loss_em: torch.Tensor
    loss_z: torch.Tensor
    loss_yz: torch.Tensor
    loss_anchor: torch.Tensor
    loss_div: torch.Tensor
    loss_macro: torch.Tensor
    loss_balance: torch.Tensor


class MALTEMLossComputer:
    def __init__(
        self,
        beta_anchor: float = 1.0,
        beta_div: float = 0.1,
        beta_macro: float = 0.5,
        beta_balance: float = 0.0,
        tau_div: float = 0.1,
        confidence_threshold: float = 0.0,
        macro_weight_mode: str = "max_prob",
        disable_anchor: bool = False,
        disable_div: bool = False,
        disable_macro: bool = False,
        disable_balance: bool = False,
        eps: float = EPS,
    ) -> None:
        self.beta_anchor = beta_anchor
        self.beta_div = beta_div
        self.beta_macro = beta_macro
        self.beta_balance = beta_balance
        self.tau_div = tau_div
        self.confidence_threshold = confidence_threshold
        self.macro_weight_mode = str(macro_weight_mode).strip().lower()
        self.disable_anchor = disable_anchor
        self.disable_div = disable_div
        self.disable_macro = disable_macro
        self.disable_balance = disable_balance
        self.eps = eps

    def macro_weights(self, p0: torch.Tensor) -> torch.Tensor:
        if self.macro_weight_mode == "none":
            return torch.ones(p0.shape[0], device=p0.device, dtype=p0.dtype)
        weights = p0.max(dim=1).values
        if self.macro_weight_mode == "threshold" and self.confidence_threshold > 0.0:
            weights = torch.where(
                weights >= self.confidence_threshold,
                weights,
                0.25 * weights,
            )
        return weights

    def em_loss(
        self,
        p0: torch.Tensor,
        q: torch.Tensor,
        prob_z_x: torch.Tensor,
        prob_y_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_pz = safe_log(prob_z_x)
        log_pyz = safe_log(prob_y_z)
        loss_z = -(q * log_pz).sum(dim=1).mean()
        loss_yz = -(q.unsqueeze(-1) * p0.unsqueeze(1) * log_pyz.unsqueeze(0)).sum(dim=(1, 2)).mean()
        loss_em = loss_z + loss_yz
        return loss_em, loss_z, loss_yz

    def anchor_loss(
        self,
        mu_target: torch.Tensor,
        mu_source: torch.Tensor,
    ) -> torch.Tensor:
        target_norm = F.normalize(mu_target, p=2, dim=1)
        source_norm = F.normalize(mu_source, p=2, dim=1)
        return ((target_norm - source_norm) ** 2).sum(dim=1).mean()

    def diversity_loss(self, nu: torch.Tensor) -> torch.Tensor:
        nu_norm = F.normalize(nu, p=2, dim=1)
        cosine = nu_norm @ nu_norm.transpose(0, 1)
        k = cosine.shape[0]
        if k < 2:
            return cosine.new_zeros(())
        mask = ~torch.eye(k, dtype=torch.bool, device=cosine.device)
        return torch.exp(cosine[mask] / self.tau_div).mean()

    def macro_consistency_loss(
        self,
        p0: torch.Tensor,
        prob_y_x: torch.Tensor,
    ) -> torch.Tensor:
        weights = self.macro_weights(p0)
        log_pt = safe_log(prob_y_x)
        per_sample = -(p0 * log_pt).sum(dim=1)
        return (per_sample * weights).sum() / weights.sum().clamp_min(self.eps)

    def balance_loss(
        self,
        prob_z_x: torch.Tensor,
        prior_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pbar = prob_z_x.mean(dim=0)
        k = pbar.shape[0]
        if prior_z is None:
            prior = torch.full((k,), 1.0 / k, device=pbar.device, dtype=pbar.dtype)
        else:
            prior = prior_z.to(device=pbar.device, dtype=pbar.dtype)
            prior = prior / prior.sum().clamp_min(self.eps)
        return F.kl_div(safe_log(pbar.unsqueeze(0)), prior.unsqueeze(0), reduction="batchmean")

    def forward(
        self,
        p0: torch.Tensor,
        q: torch.Tensor,
        prob_z_x: torch.Tensor,
        prob_y_z: torch.Tensor,
        prob_y_x: torch.Tensor,
        mu_target: torch.Tensor,
        mu_source: torch.Tensor,
        nu: torch.Tensor,
        prior_z: Optional[torch.Tensor] = None,
    ) -> MALTEMLossOutput:
        zero = prob_z_x.new_zeros(())
        loss_em, loss_z, loss_yz = self.em_loss(p0, q, prob_z_x, prob_y_z)
        loss_anchor = zero if self.disable_anchor else self.anchor_loss(mu_target, mu_source)
        loss_div = zero if self.disable_div else self.diversity_loss(nu)
        loss_macro = zero if self.disable_macro else self.macro_consistency_loss(p0, prob_y_x)
        loss_balance = (
            zero if self.disable_balance or self.beta_balance <= 0.0 else self.balance_loss(prob_z_x, prior_z)
        )
        loss_total = (
            loss_em
            + self.beta_anchor * loss_anchor
            + self.beta_div * loss_div
            + self.beta_macro * loss_macro
            + self.beta_balance * loss_balance
        )
        return MALTEMLossOutput(
            loss_total=loss_total,
            loss_em=loss_em,
            loss_z=loss_z,
            loss_yz=loss_yz,
            loss_anchor=loss_anchor,
            loss_div=loss_div,
            loss_macro=loss_macro,
            loss_balance=loss_balance,
        )
