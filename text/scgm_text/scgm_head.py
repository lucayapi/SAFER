"""Tête SCGM-G (ancres mu_y / mu_z, loss, inférence).

Official-like SCGM objective:
  L = L_CE + gamma * L_SCGM
where L_SCGM is optimized through an EM procedure:
  E-step: infer q(z_i | v_i, y_i) with Sinkhorn
  M-step: update theta, psi, phi using SGD
  v_i = normalize(E_psi(f_theta(x_i)))
theta, psi and phi must all be trainable.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def glorot(shape):
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    return (2 * init_range) * torch.rand(shape[0], shape[1]) - init_range


class SCGMHead(nn.Module):
    """Paramètres et opérations SCGM sur des features v déjà normalisées."""

    def __init__(
        self,
        hiddim: int,
        num_classes: int,
        num_subclasses: int,
    ) -> None:
        super().__init__()
        self.mu_y = nn.Parameter(glorot([num_classes, hiddim]), requires_grad=True)
        self.mu_z = nn.Parameter(glorot([num_subclasses, hiddim]), requires_grad=True)
        self.hiddim = hiddim
        self.num_classes = num_classes
        self.num_subclasses = num_subclasses
        self.criterion_cls = nn.CrossEntropyLoss()
        self._logged_macro_no_relu = False

    def scgm_parameters(self):
        return [self.mu_y, self.mu_z]

    def _macro_class_logits(
        self,
        features: torch.Tensor,
        features_norm: torch.Tensor,
        mu_y_norm: torch.Tensor,
        norm_type: str,
    ) -> torch.Tensor:
        if norm_type == "logit":
            return features_norm @ self.mu_y.t()
        if norm_type == "weight":
            return features @ mu_y_norm.t()
        if norm_type == "logit_and_weight":
            return features_norm @ mu_y_norm.t()
        if norm_type == "none":
            return features @ self.mu_y.t()
        raise NotImplementedError(f"norm_type inconnu : {norm_type!r}")

    def loss(
        self,
        logit,
        q,
        y,
        tau,
        alpha,
        beta1=1.0,
        beta2=1.0,
        beta3=1.0,
        ang_norm=False,
        norm_type="logit",
        **_ignored,
    ):
        if not self._logged_macro_no_relu:
            print(
                "[SCGM] macro logits computed without ReLU before prototype dot product.",
                flush=True,
            )
            self._logged_macro_no_relu = True

        n = logit.shape[0]
        mu_z = F.normalize(self.mu_z, p=2, dim=1)
        mu_y = F.normalize(self.mu_y, p=2, dim=1)
        logit_norm = F.normalize(logit, p=2, dim=1)

        if ang_norm is True:
            y_sample = y @ mu_y
            logit1 = F.normalize(logit_norm - y_sample, p=2, dim=1)
            logit2 = mu_z.t().unsqueeze(0) - y_sample.unsqueeze(-1)
            logit2 = F.normalize(logit2, p=2, dim=1)
            logit1 = (logit1.unsqueeze(-1) * logit2).sum(1)
            logit1 = logit1 / tau
        else:
            logit1 = logit_norm @ (mu_z.t())
            logit1 = logit1 / tau

        ls1 = self.criterion_cls(logit1, q.argmax(1))

        logit2 = (y @ mu_y) @ (mu_z.t())
        ls2_num = torch.exp(logit2)
        ls2_den = torch.exp(mu_y @ (mu_z.t()))
        ls2 = -torch.log(ls2_num / ls2_den.sum(0).view(1, -1)) * q
        ls2 = ls2.sum() / n

        logit3 = self._macro_class_logits(logit, logit_norm, mu_y, norm_type)
        ls3 = self.criterion_cls(logit3, y.argmax(1))

        ls = alpha * (beta1 * ls1 + beta2 * ls2) + beta3 * ls3
        zero = torch.tensor(0.0, device=logit.device)
        return ls, ls1, ls2, ls3, zero, zero, zero

    def pred(self, x, tau):
        x = F.normalize(x, p=2, dim=1)
        mu_z = F.normalize(self.mu_z, p=2, dim=1)
        mu_y = F.normalize(self.mu_y, p=2, dim=1)

        prob_z_given_x = torch.exp((x @ (mu_z.t())) / tau)
        prob_z_given_x = prob_z_given_x / prob_z_given_x.sum(1).view(-1, 1)

        prob_y_given_z = torch.exp((mu_z @ mu_y.t()))
        prob_y_given_z = prob_y_given_z / prob_y_given_z.sum(1).view(-1, 1)

        prob_y_given_x = prob_z_given_x @ prob_y_given_z
        return prob_y_given_x, prob_z_given_x, prob_y_given_z

    def compute_latent_sinkhorn_scores(
        self, x: torch.Tensor, y: torch.Tensor, tau: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.normalize(x, p=2, dim=1)
        mu_z = F.normalize(self.mu_z, p=2, dim=1)
        mu_y = F.normalize(self.mu_y, p=2, dim=1)

        prob_z_given_x = torch.exp((x @ (mu_z.t())) / tau)
        prob_z_given_x = prob_z_given_x / prob_z_given_x.sum(1).view(-1, 1)

        prob_y_given_z_num = torch.exp((y @ mu_y) @ (mu_z.t()))
        prob_y_given_z_den = torch.exp(mu_y @ (mu_z.t()))
        prob_y_given_z = prob_y_given_z_num / prob_y_given_z_den.sum(0).view(1, -1)

        score_for_sinkhorn = prob_z_given_x * prob_y_given_z
        return score_for_sinkhorn, prob_y_given_z, prob_z_given_x

    def forward_to_logits(self, x, y, tau=0.1, norm_type="logit"):
        x_norm = F.normalize(x, p=2, dim=1)
        mu_z = F.normalize(self.mu_z, p=2, dim=1)
        mu_y = F.normalize(self.mu_y, p=2, dim=1)

        logit1 = x_norm @ (mu_z.t())
        logit1 = logit1 / tau

        logit2 = (y @ mu_y) @ (mu_z.t())
        logit3 = self._macro_class_logits(x, x_norm, mu_y, norm_type)

        return logit1, logit2, logit3
