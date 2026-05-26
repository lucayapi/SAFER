"""Backbone texte Hugging Face (fine-tunable, differentiable forward)."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from scgm_text.pooling import pool_outputs


class TextBackbone(nn.Module):
    """Encodeur f_theta : AutoModel + mean pooling (attention_mask)."""

    def __init__(
        self,
        model_name_or_path: str,
        pooling: str = "mean",
        train_last_n_layers: Optional[int] = None,
        freeze: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.model_name_or_path = str(model_name_or_path)
        self.pooling = str(pooling).strip().lower()
        self._unfrozen_layer_count: Optional[int] = None

        if self.model_name_or_path == "__test_dummy__":
            self.embed = nn.Embedding(128, 32)
            self.hidden_size = 32
            self.model = nn.Module()
            self.model.layers = nn.ModuleList(
                [nn.Linear(32, 32) for _ in range(4)]
            )
        else:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
            )
            self.hidden_size = int(self.model.config.hidden_size)

        if freeze:
            self.freeze_all()
        elif train_last_n_layers is not None:
            self.unfreeze_last_n_layers(int(train_last_n_layers))
        elif not freeze:
            for param in self.parameters():
                param.requires_grad = True

        if (
            gradient_checkpointing
            and self.model is not None
            and hasattr(self.model, "gradient_checkpointing_enable")
        ):
            self.model.gradient_checkpointing_enable()

    def _get_transformer_layers(self) -> Optional[List[nn.Module]]:
        if self.model is None:
            return None
        if hasattr(self.model, "layers"):
            return list(self.model.layers)
        encoder = getattr(self.model, "encoder", None)
        if encoder is not None and hasattr(encoder, "layer"):
            return list(encoder.layer)
        return None

    def num_transformer_layers(self) -> Optional[int]:
        layers = self._get_transformer_layers()
        return len(layers) if layers is not None else None

    def count_trainable_transformer_layers(self) -> int:
        layers = self._get_transformer_layers()
        if not layers:
            return 0
        return sum(
            1
            for layer in layers
            if any(p.requires_grad for p in layer.parameters())
        )

    def freeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def _unfreeze_final_norms(self) -> None:
        if self.model is None:
            return
        for attr in ("norm", "final_layernorm", "ln_f"):
            module = getattr(self.model, attr, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True

    def unfreeze_last_n_layers(self, n: int) -> None:
        self.freeze_all()
        self._unfrozen_layer_count = None
        if n <= 0 or self.model is None:
            return

        layer_list = self._get_transformer_layers()
        if layer_list is None:
            print(
                "[WARN] Could not detect transformer layers; backbone stays frozen.",
                flush=True,
            )
            return

        n_layers = len(layer_list)
        if n > n_layers:
            raise ValueError(
                f"train_last_n_layers={n} exceeds total transformer layers={n_layers}"
            )

        for layer in layer_list[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        self._unfreeze_final_norms()
        self._unfrozen_layer_count = n

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.model_name_or_path == "__test_dummy__":
            last_hidden = self.embed(input_ids)
            for layer in self.model.layers:
                last_hidden = layer(last_hidden)
            return pool_outputs(last_hidden, attention_mask, mode=self.pooling)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return pool_outputs(outputs.last_hidden_state, attention_mask, mode=self.pooling)
