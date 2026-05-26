"""SCGM end-to-end: texte -> backbone f_theta -> projecteur E_psi -> tête SCGM."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from scgm_text.backbone import TextBackbone
from scgm_text.config_parsing import normalize_backbone_trainability
from scgm_text.projection import build_embedding_projector, normalize_projection_name
from scgm_text.scgm_head import SCGMHead


def _resolve_backbone_name(args: Any) -> str:
    name = getattr(args, "backbone_model_name_or_path", None) or getattr(
        args, "backbone_name", None
    )
    if not name:
        raise ValueError("backbone_name (or backbone_model_name_or_path) is required.")
    return str(name)


class SCGMTextModel(nn.Module):
    """
    Pipeline officielle-like:
      v_i = normalize(E_psi(f_theta(x_i)))
    avec theta (backbone), psi (projecteur fc/mlp), phi (prototypes SCGM).
    """

    def __init__(
        self,
        hiddim: int,
        num_classes: int,
        num_subclasses: int,
        backbone_model_name_or_path: str,
        projection: str = "linear",
        dropout: float = 0.0,
        pooling: str = "mean",
        gradient_checkpointing: bool = False,
        train_last_n_layers: Optional[int] = None,
        backbone_trainable: bool = True,
    ) -> None:
        super().__init__()
        proj_name = normalize_projection_name(projection, None)
        self.projection_name = proj_name
        self.pooling = str(pooling).strip().lower()
        self.backbone_model_name_or_path = str(backbone_model_name_or_path)

        backbone_trainable, train_last_n_layers = normalize_backbone_trainability(
            bool(backbone_trainable), train_last_n_layers
        )
        self.backbone_trainable = backbone_trainable
        self.train_last_n_layers = train_last_n_layers

        effective_gc = bool(gradient_checkpointing) and backbone_trainable
        if gradient_checkpointing and not effective_gc:
            print(
                "[SCGM] gradient_checkpointing disabled because backbone_trainable=false",
                flush=True,
            )

        self.backbone = TextBackbone(
            model_name_or_path=self.backbone_model_name_or_path,
            pooling=self.pooling,
            freeze=not backbone_trainable,
            train_last_n_layers=train_last_n_layers if backbone_trainable else None,
            gradient_checkpointing=effective_gc,
        )
        backbone_dim = self.backbone.hidden_size
        self.hiddim = int(hiddim)
        self.input_dim = int(backbone_dim)

        self.projector = build_embedding_projector(
            proj_name, self.input_dim, self.hiddim, dropout=dropout
        )
        self.head = SCGMHead(self.hiddim, num_classes, num_subclasses)
        self.num_classes = num_classes
        self.num_subclasses = num_subclasses

    @property
    def mu_y(self) -> nn.Parameter:
        return self.head.mu_y

    @property
    def mu_z(self) -> nn.Parameter:
        return self.head.mu_z

    @property
    def has_projection(self) -> bool:
        return True

    @property
    def has_trainable_backbone(self) -> bool:
        return any(p.requires_grad for p in self.backbone.parameters())

    def scgm_parameters(self):
        return self.head.scgm_parameters()

    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.has_trainable_backbone:
            with torch.no_grad():
                h = self.backbone(batch["input_ids"], batch["attention_mask"])
            h = h.detach()
        else:
            h = self.backbone(batch["input_ids"], batch["attention_mask"])
        v = self.projector(h)
        return F.normalize(v, p=2, dim=1)

    def forward(self, batch: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if isinstance(batch, torch.Tensor):
            raise ValueError("SCGM end2end expects a dict batch with input_ids and attention_mask.")
        return self.encode(batch)

    def loss(self, logit, q, y, tau, alpha, **kwargs):
        return self.head.loss(logit, q, y, tau, alpha, **kwargs)

    def pred(self, x, tau):
        return self.head.pred(x, tau)

    def compute_latent_sinkhorn_scores(self, x, y, tau):
        return self.head.compute_latent_sinkhorn_scores(x, y, tau)

    def forward_to_logits(self, x, y, tau=0.1, norm_type="logit"):
        return self.head.forward_to_logits(x, y, tau=tau, norm_type=norm_type)

    @classmethod
    def from_args(cls, args: Any) -> "SCGMTextModel":
        backbone_trainable = bool(getattr(args, "backbone_trainable", True))
        train_last_n_layers = getattr(args, "train_last_n_layers", None)
        gc = bool(getattr(args, "effective_gradient_checkpointing", getattr(args, "gradient_checkpointing", False)))
        return cls(
            hiddim=int(getattr(args, "hiddim", 128)),
            num_classes=int(args.n_class),
            num_subclasses=int(args.n_subclass),
            backbone_model_name_or_path=_resolve_backbone_name(args),
            projection=getattr(args, "projection", "linear"),
            dropout=float(getattr(args, "dropout", 0.0)),
            pooling=getattr(args, "pooling", "mean"),
            gradient_checkpointing=gc,
            train_last_n_layers=train_last_n_layers,
            backbone_trainable=backbone_trainable,
        )
