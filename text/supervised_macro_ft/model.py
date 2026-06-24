"""Modèle CE : TextBackbone + Linear(num_classes)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from scgm_text.backbone import TextBackbone
from scgm_text.config_parsing import normalize_backbone_trainability


class SupervisedMacroModel(nn.Module):
    """h = pool(f_theta(u)) ; logits = W h + b (sans L2 avant la tête)."""

    def __init__(
        self,
        *,
        backbone_name: str,
        num_classes: int = 4,
        pooling: str = "mean",
        backbone_trainable: bool = True,
        train_last_n_layers: Optional[int] = None,
        gradient_checkpointing: bool = False,
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

        effective_gc = bool(gradient_checkpointing) and backbone_trainable
        self.backbone = TextBackbone(
            model_name_or_path=self.backbone_name,
            pooling=self.pooling,
            freeze=not backbone_trainable,
            train_last_n_layers=train_last_n_layers if backbone_trainable else None,
            gradient_checkpointing=effective_gc,
        )
        hidden = int(self.backbone.hidden_size)
        self.classifier = nn.Linear(hidden, self.num_classes)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask)

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encode(input_ids, attention_mask)
        return self.classifier(h)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
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
