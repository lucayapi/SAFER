"""Tête SCGM-G partagée (ancres mu_y / mu_z, loss, inférence)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scgm_text.distillation import DistillKL


def glorot(shape):
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    return (2 * init_range) * torch.rand(shape[0], shape[1]) - init_range


class SCGMHead(nn.Module):
    """Paramètres et opérations SCGM sur des features h déjà encodées."""

    def __init__(
        self,
        hiddim: int,
        num_classes: int,
        num_subclasses: int,
        kd_t: float = 4.0,
    ) -> None:
        super().__init__()
        self.mu_y = nn.Parameter(glorot([num_classes, hiddim]), requires_grad=True)
        self.mu_z = nn.Parameter(glorot([num_subclasses, hiddim]), requires_grad=True)
        self.hiddim = hiddim
        self.num_classes = num_classes
        self.num_subclasses = num_subclasses
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_div = DistillKL(kd_t)

    def scgm_parameters(self):
        return [self.mu_y, self.mu_z]

    def loss(
        self,
        logit,
        q,
        y,
        tau,
        alpha,
        logit_t1=None,
        logit_t2=None,
        logit_t3=None,
        beta1=1.0,
        beta2=1.0,
        beta3=1.0,
        ang_norm=False,
        norm_type="logit",
        kd_t: Optional[float] = None,
    ):
        if kd_t is not None:
            self.criterion_div.T = float(kd_t)

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

        if norm_type == "logit":
            logit3 = (F.relu(logit_norm)) @ (self.mu_y.t())
        elif norm_type == "weight":
            logit3 = (F.relu(logit)) @ (mu_y.t())
        elif norm_type == "logit_and_weight":
            logit3 = (F.relu(logit_norm)) @ (mu_y.t())
        elif norm_type == "none":
            logit3 = (F.relu(logit)) @ (self.mu_y.t())
        else:
            raise NotImplementedError

        ls3 = self.criterion_cls(logit3, y.argmax(1))

        if beta1 == 1.0 or logit_t1 is None:
            ls_div1 = torch.tensor(0.0, device=logit.device)
        else:
            ls_div1 = self.criterion_div(logit1, logit_t1)

        if beta2 == 1.0 or logit_t2 is None:
            ls_div2 = torch.tensor(0.0, device=logit.device)
        else:
            ls_div2 = self.criterion_div(logit2, logit_t2)

        if beta3 == 1.0 or logit_t3 is None:
            ls_div3 = torch.tensor(0.0, device=logit.device)
        else:
            ls_div3 = self.criterion_div(logit3, logit_t3)

        ls = (
            alpha * (beta1 * ls1 + beta2 * ls2)
            + beta3 * ls3
            + (1.0 - beta1) * ls_div1
            + (1.0 - beta2) * ls_div2
            + (1.0 - beta3) * ls_div3
        )
        return ls, ls1, ls2, ls3, ls_div1, ls_div2, ls_div3

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

        if norm_type == "logit":
            logit3 = (F.relu(x_norm)) @ (self.mu_y.t())
        elif norm_type == "weight":
            logit3 = (F.relu(x)) @ (mu_y.t())
        elif norm_type == "logit_and_weight":
            logit3 = (F.relu(x_norm)) @ (mu_y.t())
        elif norm_type == "none":
            logit3 = (F.relu(x)) @ (mu_y.t())
        else:
            raise NotImplementedError

        return logit1, logit2, logit3
