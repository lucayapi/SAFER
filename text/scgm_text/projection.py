"""Projecteurs partagés SCGM texte (backbone → espace des ancres)."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import torch
import torch.nn as nn

ProjectionName = Literal["linear", "ln_gelu", "residual", "mlp_sklearn", "mlp"]
# mlp (ReLU) : legacy SCGM. mlp_sklearn : baseline 07 (256→128). macro FT : linear | ln_gelu | residual | mlp_sklearn.

SKLEARN_MLP_HIDDEN = 256
SKLEARN_MLP_OUT_DIM = 128


class ResidualProjector(nn.Module):
    """r = LayerNorm(h + α·g(h)) ; z = Linear(d_in → hiddim)(r), g : d_in → bottleneck → d_in."""

    def __init__(
        self,
        input_dim: int,
        hiddim: int,
        *,
        bottleneck: int = 256,
        alpha: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.g = nn.Sequential(
            nn.Linear(input_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(bottleneck, input_dim),
        )
        self.norm = nn.LayerNorm(input_dim)
        self.out_proj = nn.Linear(input_dim, hiddim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        r = self.norm(h + self.alpha * self.g(h))
        return self.out_proj(r)


def normalize_projection_name(projection: Optional[str], with_mlp: Optional[bool] = None) -> str:
    """
    ``projection`` ∈ {fc, linear, ln_gelu, residual, mlp_sklearn, mlp}.
    ``fc`` ≡ ``linear``. ``sklearn_mlp`` ≡ ``mlp_sklearn``. ``mlp`` = legacy SCGM (ReLU).
    """
    if projection is not None and str(projection).strip():
        p = str(projection).strip().lower()
        if p in ("fc", "linear"):
            return "linear"
        if p == "ln_gelu":
            return "ln_gelu"
        if p == "residual":
            return "residual"
        if p in ("mlp_sklearn", "sklearn_mlp"):
            return "mlp_sklearn"
        if p == "mlp":
            return "mlp"
        if p == "identity":
            raise ValueError(
                "projection=identity is not supported on the official SCGM end2end pipeline. "
                "Use projection=fc (linear) or mlp with hiddim < backbone dim."
            )
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
    *,
    proj_hidden: Optional[int] = None,
    proj_bottleneck: Optional[int] = None,
    proj_alpha: float = 0.1,
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
    if p == "ln_gelu":
        hidden = int(proj_hidden if proj_hidden is not None else min(input_dim, 512))
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, hiddim))
        return nn.Sequential(*layers)
    if p == "residual":
        bottleneck = int(proj_bottleneck if proj_bottleneck is not None else 256)
        return ResidualProjector(
            input_dim,
            hiddim,
            bottleneck=bottleneck,
            alpha=float(proj_alpha),
            dropout=dropout,
        )
    if p == "mlp_sklearn":
        hidden = int(proj_hidden if proj_hidden is not None else SKLEARN_MLP_HIDDEN)
        out_dim = int(hiddim if hiddim == SKLEARN_MLP_OUT_DIM else SKLEARN_MLP_OUT_DIM)
        layers = [nn.Linear(input_dim, hidden), nn.ReLU()]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, out_dim))
        return nn.Sequential(*layers)
    # legacy SCGM : mlp avec ReLU
    layers = [nn.Linear(input_dim, input_dim), nn.ReLU()]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(input_dim, hiddim))
    return nn.Sequential(*layers)
