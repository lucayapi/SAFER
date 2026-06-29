"""Loss de préservation des similarités cosinus entre h (Qwen) et z (projeté)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def similarity_preservation_loss(
    h: torch.Tensor,
    z: torch.Tensor,
    *,
    remove_diag: bool = True,
) -> torch.Tensor:
    """
    L_geo = MSE(S_z, S_h) sur similarités cosinus.

    h : embeddings backbone [B, d_h]
    z : embeddings projetés [B, d_z]
    """
    if h.ndim != 2 or z.ndim != 2 or h.shape[0] != z.shape[0]:
        raise ValueError(f"h et z doivent être [B, d] avec même B : {h.shape}, {z.shape}")
    batch_size = int(h.shape[0])
    if batch_size < 2:
        return h.new_zeros(())

    h_norm = F.normalize(h.float(), p=2, dim=1)
    z_norm = F.normalize(z.float(), p=2, dim=1)
    s_h = h_norm @ h_norm.T
    s_z = z_norm @ z_norm.T

    if remove_diag:
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=h.device)
        return F.mse_loss(s_z[mask], s_h.detach()[mask])
    return F.mse_loss(s_z, s_h.detach())
