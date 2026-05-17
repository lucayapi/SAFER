"""Modèle SCGM texte unifié : backbone HF ou embeddings pré-calculés."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from scgm_text.backbone import TextBackbone
from scgm_text.projection import build_embedding_projector, normalize_projection_name
from scgm_text.scgm_head import SCGMHead

InputMode = str  # "text" | "precomputed_embeddings"


class SCGMTextModel(nn.Module):
    """
    SCGM-G pour texte.

    - ``input_mode="text"`` : ``h = f_theta(x)`` (backbone HF), puis projection optionnelle.
    - ``input_mode="precomputed_embeddings"`` : ``h = projection(e)`` ou ``h = e`` si identity.

    ``projection="identity"`` signifie l'absence de projecteur **additionnel**, pas des embeddings figés :
    en mode texte avec ``freeze_backbone=False``, theta est fine-tuné par la loss SCGM.
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
        kd_t: float = 4.0,
        input_mode: str = "precomputed_embeddings",
        backbone_model_name_or_path: Optional[str] = None,
        pooling: str = "mean",
        freeze_backbone: bool = False,
        train_last_n_layers: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_mode = str(input_mode).strip().lower()
        if self.input_mode not in ("text", "precomputed_embeddings"):
            raise ValueError(f"input_mode inconnu : {input_mode!r}")

        if with_mlp is not None:
            proj_name = normalize_projection_name(None, with_mlp)
        else:
            proj_name = normalize_projection_name(projection, None)
        self.projection_name = proj_name
        self.pooling = str(pooling).strip().lower()
        self.freeze_backbone = bool(freeze_backbone)

        self.backbone: Optional[TextBackbone] = None
        if self.input_mode == "text":
            if not backbone_model_name_or_path:
                raise ValueError("input_mode=text exige backbone_model_name_or_path.")
            self.backbone = TextBackbone(
                model_name_or_path=backbone_model_name_or_path,
                pooling=self.pooling,
                train_last_n_layers=train_last_n_layers,
                freeze=freeze_backbone,
            )
            backbone_dim = self.backbone.hidden_size
        else:
            backbone_dim = int(input_dim)

        if proj_name == "identity":
            if int(hiddim) != int(backbone_dim):
                hiddim = int(backbone_dim)
        self.hiddim = int(hiddim)
        self.input_dim = int(backbone_dim if self.input_mode == "text" else input_dim)

        self.projector = build_embedding_projector(
            proj_name, self.input_dim, self.hiddim, dropout=dropout
        )
        self.head = SCGMHead(self.hiddim, num_classes, num_subclasses, kd_t=kd_t)
        self.num_classes = num_classes
        self.num_subclasses = num_subclasses
        self.criterion_cls = self.head.criterion_cls
        self.criterion_div = self.head.criterion_div

    @property
    def mu_y(self) -> nn.Parameter:
        return self.head.mu_y

    @property
    def mu_z(self) -> nn.Parameter:
        return self.head.mu_z

    @property
    def has_projection(self) -> bool:
        return self.projection_name != "identity"

    @property
    def has_trainable_backbone(self) -> bool:
        if self.backbone is None:
            return False
        return any(p.requires_grad for p in self.backbone.parameters())

    def scgm_parameters(self):
        return self.head.scgm_parameters()

    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.input_mode == "text":
            if self.backbone is None:
                raise RuntimeError("backbone manquant en mode text.")
            e = self.backbone.encode(
                batch["input_ids"],
                batch["attention_mask"],
            )
        else:
            e = batch["embeddings"]
        if self.projection_name == "identity":
            return e
        return self.projector(e)

    def forward(self, batch: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if isinstance(batch, torch.Tensor):
            if self.input_mode != "precomputed_embeddings":
                raise ValueError(
                    "Tensor brut : utiliser un dict batch en input_mode=text."
                )
            batch = {"embeddings": batch}
        h = self.encode(batch)
        return F.normalize(h, p=2, dim=1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Compatibilité SCGMEmbeddingNet : tenseur d'embeddings pré-calculés."""
        if self.input_mode == "text":
            raise ValueError("embed(tensor) non supporté en input_mode=text ; utiliser forward(batch).")
        if self.projection_name == "identity":
            return x
        return self.projector(x)

    def loss(self, logit, q, y, tau, alpha, **kwargs):
        return self.head.loss(logit, q, y, tau, alpha, **kwargs)

    def pred(self, x, tau):
        return self.head.pred(x, tau)

    def compute_latent_sinkhorn_scores(self, x, y, tau):
        return self.head.compute_latent_sinkhorn_scores(x, y, tau)

    def forward_to_logits(self, x, y, tau=0.1, norm_type="logit"):
        return self.head.forward_to_logits(x, y, tau=tau, norm_type=norm_type)

    @classmethod
    def from_args(cls, args: Any, input_dim: int) -> "SCGMTextModel":
        input_mode = getattr(args, "input_mode", "precomputed_embeddings")
        hiddim = int(getattr(args, "hiddim", 128))
        projection = getattr(args, "projection", "identity")
        if projection == "identity" and input_mode == "precomputed_embeddings":
            hiddim = int(input_dim)
        elif projection == "identity" and input_mode == "text":
            pass
        return cls(
            input_dim=input_dim,
            hiddim=hiddim,
            num_classes=int(args.n_class),
            num_subclasses=int(args.n_subclass),
            projection=projection,
            kd_t=float(getattr(args, "kd_t", 4.0)),
            input_mode=input_mode,
            backbone_model_name_or_path=getattr(args, "backbone_model_name_or_path", None),
            pooling=getattr(args, "pooling", "mean"),
            freeze_backbone=bool(getattr(args, "freeze_backbone", False)),
            train_last_n_layers=getattr(args, "train_last_n_layers", None),
        )
