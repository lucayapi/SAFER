from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from malt_text.utils import EPS, safe_log
from sinkhornknopp import optimize_l_sk


@dataclass
class MALTLossOutput:
    loss_total: torch.Tensor
    loss_softmacro: torch.Tensor
    loss_latent: torch.Tensor
    loss_anchor: torch.Tensor
    loss_div: torch.Tensor


class MALTLossComputer:
    def __init__(
        self,
        beta_latent: float = 1.0,
        beta_anchor: float = 1.0,
        beta_div: float = 0.1,
        tau_div: float = 0.1,
        confidence_threshold: float = 0.0,
        latent_loss_mode: str = "marginal",
        use_sinkhorn: bool = True,
        sinkhorn_lmd: float = 25.0,
        disable_softmacro: bool = False,
        disable_latent: bool = False,
        disable_anchor: bool = False,
        disable_div: bool = False,
    ) -> None:
        self.beta_latent = beta_latent
        self.beta_anchor = beta_anchor
        self.beta_div = beta_div
        self.tau_div = tau_div
        self.confidence_threshold = confidence_threshold
        self.latent_loss_mode = latent_loss_mode
        self.use_sinkhorn = use_sinkhorn
        self.sinkhorn_lmd = sinkhorn_lmd
        self.disable_softmacro = disable_softmacro
        self.disable_latent = disable_latent
        self.disable_anchor = disable_anchor
        self.disable_div = disable_div

    def confidence_weights(self, p0: torch.Tensor) -> torch.Tensor:
        weights = p0.max(dim=1).values
        if self.confidence_threshold > 0.0:
            weights = torch.where(
                weights >= self.confidence_threshold,
                weights,
                0.25 * weights,
            )
        return weights

    def soft_macro_loss(
        self,
        p0: torch.Tensor,
        pt: torch.Tensor,
    ) -> torch.Tensor:
        weights = self.confidence_weights(p0)
        log_pt = safe_log(pt)
        per_sample = -(p0 * log_pt).sum(dim=1)
        return (per_sample * weights).sum() / weights.sum().clamp_min(EPS)

    def latent_marginal_loss(
        self,
        p0: torch.Tensor,
        prob_z_x: torch.Tensor,
        prob_y_z: torch.Tensor,
    ) -> torch.Tensor:
        log_pz = safe_log(prob_z_x)
        log_pyz = safe_log(prob_y_z)
        soft_macro_score = torch.einsum("bm,km->bk", p0, log_pyz)
        log_terms = log_pz + soft_macro_score
        return -torch.logsumexp(log_terms, dim=1).mean()

    def latent_sinkhorn_loss(
        self,
        p0: torch.Tensor,
        prob_z_x: torch.Tensor,
        prob_y_z: torch.Tensor,
    ) -> torch.Tensor:
        log_pyz = safe_log(prob_y_z)
        soft_macro_score = torch.einsum("bm,km->bk", p0, log_pyz)
        q_raw = prob_z_x * torch.exp(soft_macro_score)
        q_np, _ = optimize_l_sk(q_raw.detach().cpu().numpy(), self.sinkhorn_lmd)
        q = torch.from_numpy(q_np).to(prob_z_x.device, dtype=prob_z_x.dtype)
        loss_q = -(q * safe_log(prob_z_x)).sum(dim=1).mean()
        loss_yz = -(q.unsqueeze(-1) * p0.unsqueeze(1) * log_pyz.unsqueeze(0)).sum(dim=(1, 2)).mean()
        return loss_q + loss_yz

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

    def forward(
        self,
        p0: torch.Tensor,
        prob_z_x: torch.Tensor,
        prob_y_z: torch.Tensor,
        pt: torch.Tensor,
        mu_target: torch.Tensor,
        mu_source: torch.Tensor,
        nu: torch.Tensor,
    ) -> MALTLossOutput:
        zero = pt.new_zeros(())
        loss_softmacro = zero if self.disable_softmacro else self.soft_macro_loss(p0, pt)
        if self.disable_latent:
            loss_latent = zero
        elif self.latent_loss_mode == "sinkhorn" and self.use_sinkhorn:
            loss_latent = self.latent_sinkhorn_loss(p0, prob_z_x, prob_y_z)
        else:
            loss_latent = self.latent_marginal_loss(p0, prob_z_x, prob_y_z)
        loss_anchor = zero if self.disable_anchor else self.anchor_loss(mu_target, mu_source)
        loss_div = zero if self.disable_div else self.diversity_loss(nu)
        loss_total = (
            loss_softmacro
            + self.beta_latent * loss_latent
            + self.beta_anchor * loss_anchor
            + self.beta_div * loss_div
        )
        return MALTLossOutput(
            loss_total=loss_total,
            loss_softmacro=loss_softmacro,
            loss_latent=loss_latent,
            loss_anchor=loss_anchor,
            loss_div=loss_div,
        )
