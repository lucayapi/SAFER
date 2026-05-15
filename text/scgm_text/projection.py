"""Projecteurs partagés SCGM texte / MALT (backbone → espace des ancres)."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import torch.nn as nn

ProjectionName = Literal["identity", "linear", "mlp"]


def normalize_projection_name(projection: Optional[str], with_mlp: Optional[bool] = None) -> str:
    """
    Anciens checkpoints : seulement ``with_mlp`` (bool).
    Nouveaux : ``projection`` ∈ {identity, linear, mlp}.
    """
    if projection is not None and str(projection).strip():
        p = str(projection).strip().lower()
        if p in ("identity", "linear", "mlp"):
            return p
        raise ValueError(f"projection inconnu : {projection!r}")
    if with_mlp is None:
        return "mlp"
    return "mlp" if bool(with_mlp) else "linear"


def projection_from_checkpoint_args(args: Optional[Dict[str, Any]]) -> str:
    """Lit ``projection`` ou migre depuis l’ancien ``with_mlp``."""
    if not args:
        return "mlp"
    raw = args.get("projection")
    if raw is not None and str(raw).strip():
        return normalize_projection_name(str(raw), None)
    return normalize_projection_name(None, args.get("with_mlp", True))


def build_embedding_projector(
    projection: str,
    input_dim: int,
    hiddim: int,
    dropout: float = 0.0,
) -> nn.Module:
    p = normalize_projection_name(projection, None)
    if p == "identity":
        if int(hiddim) != int(input_dim):
            raise ValueError(
                f"projection=identity exige hiddim==input_dim (hiddim={hiddim}, input_dim={input_dim})."
            )
        return nn.Identity()
    if p == "linear":
        return nn.Linear(input_dim, hiddim)
    layers = [nn.Linear(input_dim, input_dim), nn.ReLU()]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(input_dim, hiddim))
    return nn.Sequential(*layers)
