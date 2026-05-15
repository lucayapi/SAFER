from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scgm_text.projection import build_embedding_projector, normalize_projection_name


def glorot(shape):
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    return (2 * init_range) * torch.rand(shape[0], shape[1]) - init_range


class SCGMEmbeddingNet(nn.Module):
    """SCGM-G head for fixed text embeddings.

    ``projection`` : ``identity`` (backbone natif, ``hiddim == input_dim``),
    ``linear`` (un seul linéaire), ``mlp`` (2 couches + ReLU).
    """

    def __init__(
        self,
        input_dim: int,
        hiddim: int,
        num_classes: int,
        num_subclasses: int,
        projection: str = "identity",
        dropout: float = 0.0,
        with_mlp: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if with_mlp is not None:
            proj_name = normalize_projection_name(None, with_mlp)
        else:
            proj_name = normalize_projection_name(projection, None)
        self.projection_name = proj_name
        self.projector = build_embedding_projector(proj_name, input_dim, hiddim, dropout=dropout)

        self.mu_y = nn.Parameter(glorot([num_classes, hiddim]), requires_grad=True)
        self.mu_z = nn.Parameter(glorot([num_subclasses, hiddim]), requires_grad=True)
        self.hiddim = hiddim
        self.num_classes = num_classes
        self.num_subclasses = num_subclasses
        self.criterion_cls = nn.CrossEntropyLoss()

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.embed(x), p=2, dim=1)

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
    ):
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

        ls_div1 = 0.0 if beta1 == 1.0 else 0.0
        ls_div2 = 0.0 if beta2 == 1.0 else 0.0
        ls_div3 = 0.0 if beta3 == 1.0 else 0.0

        ls = alpha * (beta1 * ls1 + beta2 * ls2) + beta3 * ls3
        return ls, ls1, ls2, ls3, ls_div1, ls_div2, ls_div3

    def pred(self, x, tau):
        x = F.normalize(x, p=2, dim=1)
        mu_z = F.normalize(self.mu_z, p=2, dim=1)
        mu_y = F.normalize(self.mu_y, p=2, dim=1)

        prob_z_x = torch.exp((x @ (mu_z.t())) / tau)
        prob_z_x = prob_z_x / prob_z_x.sum(1).view(-1, 1)

        prob_y_z = torch.exp((mu_z @ mu_y.t()))
        prob_y_z = prob_y_z / prob_y_z.sum(1).view(-1, 1)

        prob_y_x = prob_z_x @ prob_y_z
        return prob_y_x, prob_z_x, prob_y_z

    def forward_to_prob(self, x, y, tau):
        x = F.normalize(x, p=2, dim=1)
        mu_z = F.normalize(self.mu_z, p=2, dim=1)
        mu_y = F.normalize(self.mu_y, p=2, dim=1)

        prob_z_x = torch.exp((x @ (mu_z.t())) / tau)
        prob_z_x = prob_z_x / prob_z_x.sum(1).view(-1, 1)

        prob_y_z_num = torch.exp((y @ mu_y) @ (mu_z.t()))
        prob_y_z_den = torch.exp(mu_y @ (mu_z.t()))
        prob_y_z = prob_y_z_num / prob_y_z_den.sum(0).view(1, -1)

        prob_y_x = prob_z_x * prob_y_z
        return prob_y_x, prob_y_z, prob_z_x
