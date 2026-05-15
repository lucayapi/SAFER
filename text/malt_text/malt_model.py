from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scgm_text.projection import build_embedding_projector, normalize_projection_name
from scgm_text.scgm_embedding_model import glorot


class MALTTargetModel(nn.Module):
    """Target MALT model with one global bank of K latent motifs."""

    def __init__(
        self,
        input_dim: int,
        hiddim: int,
        num_classes: int,
        num_subclasses: int,
        projection: str = "identity",
        dropout: float = 0.0,
        freeze_projector: bool = False,
        with_mlp: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if with_mlp is not None:
            proj_name = normalize_projection_name(None, with_mlp)
        else:
            proj_name = normalize_projection_name(projection, None)
        self.projection_name = proj_name
        self.projector = build_embedding_projector(proj_name, input_dim, hiddim, dropout)

        self.mu_y = nn.Parameter(glorot([num_classes, hiddim]), requires_grad=True)
        self.nu = nn.Parameter(glorot([num_subclasses, hiddim]), requires_grad=True)
        self.hiddim = hiddim
        self.num_classes = num_classes
        self.num_subclasses = num_subclasses
        self.freeze_projector = freeze_projector
        if freeze_projector:
            for parameter in self.projector.parameters():
                parameter.requires_grad = False

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.embed(x), p=2, dim=1)

    def macro_probs(self, features: torch.Tensor, mu_y: torch.Tensor, tau: float) -> torch.Tensor:
        features_norm = F.normalize(features, p=2, dim=1)
        mu_norm = F.normalize(mu_y, p=2, dim=1)
        logits = features_norm @ mu_norm.transpose(0, 1)
        return torch.softmax(logits / tau, dim=1)

    def latent_probs(self, features: torch.Tensor, tau: float) -> torch.Tensor:
        features_norm = F.normalize(features, p=2, dim=1)
        nu_norm = F.normalize(self.nu, p=2, dim=1)
        logits = features_norm @ nu_norm.transpose(0, 1)
        return torch.softmax(logits / tau, dim=1)

    def macro_given_latent(self, tau: float) -> torch.Tensor:
        nu_norm = F.normalize(self.nu, p=2, dim=1)
        mu_norm = F.normalize(self.mu_y, p=2, dim=1)
        logits = nu_norm @ mu_norm.transpose(0, 1)
        return torch.softmax(logits / tau, dim=1)

    def marginal_macro(self, features: torch.Tensor, tau_z: float, tau_yz: float) -> torch.Tensor:
        prob_z_x = self.latent_probs(features, tau_z)
        prob_y_z = self.macro_given_latent(tau_yz)
        return prob_z_x @ prob_y_z
