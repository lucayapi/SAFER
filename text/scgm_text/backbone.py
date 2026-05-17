"""Backbone texte Hugging Face (fine-tunable)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from scgm_text.pooling import pool_outputs


class TextBackbone(nn.Module):
    """Encodeur ``f_theta`` : AutoModel + pooling."""

    def __init__(
        self,
        model_name_or_path: str,
        pooling: str = "mean",
        train_last_n_layers: Optional[int] = None,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.model_name_or_path = str(model_name_or_path)
        self.pooling = str(pooling).strip().lower()

        if self.model_name_or_path == "__test_dummy__":
            self.embed = nn.Embedding(128, 32)
            self.hidden_size = 32
            self.model = None
        else:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(self.model_name_or_path)
            self.hidden_size = int(self.model.config.hidden_size)

        if freeze:
            self.freeze_all()
        if train_last_n_layers is not None:
            self.unfreeze_last_n_layers(int(train_last_n_layers))

    def freeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_last_n_layers(self, n: int) -> None:
        self.freeze_all()
        if n <= 0:
            return
        layers = getattr(self.model, "encoder", None)
        if layers is not None and hasattr(layers, "layer"):
            layer_list = list(layers.layer)
        elif self.model is not None and hasattr(self.model, "layers"):
            layer_list = list(self.model.layers)
        else:
            for param in self.parameters():
                param.requires_grad = True
            return
        for layer in layer_list[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        if hasattr(self.model, "embeddings"):
            pass
        if hasattr(self.model, "pooler") and self.model.pooler is not None:
            for param in self.model.pooler.parameters():
                param.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.model is None:
            last_hidden = self.embed(input_ids)
            return pool_outputs(last_hidden, attention_mask, mode=self.pooling)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state
        return pool_outputs(last_hidden, attention_mask, mode=self.pooling)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(input_ids, attention_mask)
