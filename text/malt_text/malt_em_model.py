"""MALT-EM target model: SCGM-like EM on transferred soft macro responsibilities."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from malt_text.malt_model import MALTTargetModel


class MALTEMTargetModel(MALTTargetModel):
    """Target model for MALT-EM (inherits projection, mu_y, nu from MALTTargetModel)."""

    def compute_all_probs(
        self,
        x: torch.Tensor,
        tau_z: float,
        tau_yz: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        features : (N, d)
        prob_z_x : (N, K)  latent responsibilities p_t(z|x)
        prob_y_z : (K, C)  p_t(y|z)
        prob_y_x : (N, C)  marginal macro p_t(y|x)
        """
        features = self.forward(x)
        prob_z_x = self.latent_probs(features, tau_z)
        prob_y_z = self.macro_given_latent(tau_yz)
        prob_y_x = prob_z_x @ prob_y_z
        return {
            "features": features,
            "prob_z_x": prob_z_x,
            "prob_y_z": prob_y_z,
            "prob_y_x": prob_y_x,
        }
