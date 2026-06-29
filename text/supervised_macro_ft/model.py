"""Modèle CE : TextBackbone → projecteur ψ → tête linear (num_classes)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from scgm_text.backbone import TextBackbone
from scgm_text.config_parsing import normalize_backbone_trainability
from scgm_text.projection import build_embedding_projector, normalize_projection_name


def model_kwargs_from_cfg(model_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Construit les kwargs SupervisedMacroModel depuis la section YAML ``model``."""
    projection = model_cfg.get("projection", "mlp")
    return {
        "backbone_name": str(model_cfg["backbone_name"]),
        "num_classes": int(model_cfg.get("n_classes", 4)),
        "pooling": str(model_cfg.get("pooling", "mean")),
        "backbone_trainable": bool(model_cfg.get("backbone_trainable", True)),
        "train_last_n_layers": model_cfg.get("train_last_n_layers"),
        "gradient_checkpointing": bool(model_cfg.get("gradient_checkpointing", False)),
        "projection": projection,
        "hiddim": int(model_cfg.get("hiddim", 128)),
        "dropout": float(model_cfg.get("dropout", 0.1)),
    }


class SupervisedMacroModel(nn.Module):
    """
    h = pool(f_theta(u)) ; z = ψ(h) ; logits = W z + b.

    Mode legacy (sans projecteur) : projection=None → z = h, Linear(backbone_dim, n_classes).
    """

    def __init__(
        self,
        *,
        backbone_name: str,
        num_classes: int = 4,
        pooling: str = "mean",
        backbone_trainable: bool = True,
        train_last_n_layers: Optional[int] = None,
        gradient_checkpointing: bool = False,
        projection: Optional[str] = "mlp",
        hiddim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        backbone_trainable, train_last_n_layers = normalize_backbone_trainability(
            bool(backbone_trainable), train_last_n_layers
        )
        self.backbone_name = str(backbone_name)
        self.pooling = str(pooling).strip().lower()
        self.backbone_trainable = backbone_trainable
        self.train_last_n_layers = train_last_n_layers
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)

        self.use_projector = projection is not None and str(projection).strip().lower() not in (
            "none",
            "null",
            "",
        )
        if self.use_projector:
            self.projection_name = normalize_projection_name(str(projection), None)
            self.hiddim = int(hiddim)
        else:
            self.projection_name = "legacy"
            self.hiddim = int(hiddim)

        effective_gc = bool(gradient_checkpointing) and backbone_trainable
        self.backbone = TextBackbone(
            model_name_or_path=self.backbone_name,
            pooling=self.pooling,
            freeze=not backbone_trainable,
            train_last_n_layers=train_last_n_layers if backbone_trainable else None,
            gradient_checkpointing=effective_gc,
        )
        backbone_dim = int(self.backbone.hidden_size)

        if self.use_projector:
            self.projector = build_embedding_projector(
                self.projection_name,
                backbone_dim,
                self.hiddim,
                dropout=self.dropout,
            )
            self.classifier = nn.Linear(self.hiddim, self.num_classes)
        else:
            self.projector = nn.Identity()
            self.hiddim = backbone_dim
            self.classifier = nn.Linear(backbone_dim, self.num_classes)

    @property
    def has_trainable_backbone(self) -> bool:
        return any(p.requires_grad for p in self.backbone.parameters())

    def _encode_backbone(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.has_trainable_backbone:
            with torch.no_grad():
                h = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            return h.detach()
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask)

    def encode_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """z = ψ(h) à partir d'embeddings backbone déjà calculés."""
        return self.projector(hidden)

    def forward_logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        z = self.encode_from_hidden(hidden)
        return self.classifier(z)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Retourne z = ψ(h) (espace adapté pour export / BERTopic)."""
        h = self._encode_backbone(input_ids, attention_mask)
        return self.projector(h)

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encode(input_ids, attention_mask)
        return self.classifier(z)

    def forward_with_latents(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retourne (logits, z, h) en un seul forward."""
        if "hidden" in batch:
            h = batch["hidden"]
            z = self.encode_from_hidden(h)
            logits = self.classifier(z)
            return logits, z, h
        h = self._encode_backbone(batch["input_ids"], batch["attention_mask"])
        z = self.projector(h)
        logits = self.classifier(z)
        return logits, z, h

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        if "hidden" in batch:
            return self.forward_logits_from_hidden(batch["hidden"])
        return self.forward_logits(batch["input_ids"], batch["attention_mask"])

    @torch.no_grad()
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward_logits(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)
        return probs, logits

    def prediction_stats(self, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        confidence = probs.max(dim=-1).values
        sort_p = torch.sort(probs, dim=-1).values
        margin = (
            sort_p[:, -1] - sort_p[:, -2]
            if probs.shape[-1] >= 2
            else torch.zeros(len(probs), device=probs.device)
        )
        entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=-1)
        return confidence, margin, entropy
