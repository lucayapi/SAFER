"""Encodeur HF unifié pour les méthodes contrastives.

Basé sur ``TextBackbone`` (scgm_text) avec projecteur optionnel partagé avec
supervised_macro_ft (linear | mlp_sklearn).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from scgm_text.backbone import TextBackbone
from scgm_text.config_parsing import normalize_backbone_trainability
from scgm_text.projection import SKLEARN_MLP_OUT_DIM, build_embedding_projector
from supervised_macro_ft.model import validate_macro_ft_projection

CONTRASTIVE_ENCODER_CKPT = "contrastive_encoder.pt"


@dataclass
class EncoderConfig:
    backbone_name: str
    max_seq_length: int = 256
    backbone_trainable: bool = False
    train_last_n_layers: Optional[int] = None
    cache_backbone_embeddings: bool = True
    use_projector: bool = True
    projection: str = "linear"
    hiddim: int = 128


class ContrastiveEncoder(nn.Module):
    """Backbone texte HF + projecteur optionnel."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        bb_trainable, train_last = normalize_backbone_trainability(
            bool(cfg.backbone_trainable),
            cfg.train_last_n_layers,
        )
        self.backbone_trainable = bb_trainable
        self.train_last_n_layers = train_last
        self.cache_backbone_embeddings = bool(cfg.cache_backbone_embeddings) and not bb_trainable
        self.max_seq_length = int(cfg.max_seq_length)

        freeze = not bb_trainable
        self.backbone = TextBackbone(
            cfg.backbone_name,
            pooling="mean",
            train_last_n_layers=train_last,
            freeze=freeze,
            gradient_checkpointing=bb_trainable,
        )
        in_dim = int(self.backbone.hidden_size)

        self.tokenizer = self._load_tokenizer(cfg.backbone_name)

        self.projection_name: Optional[str]
        self.projector: Optional[nn.Module]
        out_dim = in_dim
        if cfg.use_projector:
            proj_name = validate_macro_ft_projection(cfg.projection)
            self.projection_name = proj_name
            self.projector = build_embedding_projector(
                proj_name,
                in_dim,
                int(cfg.hiddim),
            )
            if proj_name == "mlp_sklearn":
                out_dim = SKLEARN_MLP_OUT_DIM
            else:
                out_dim = int(cfg.hiddim)
        else:
            self.projection_name = None
            self.projector = None

        self.embedding_dim = int(out_dim)

    @staticmethod
    def _load_tokenizer(backbone_name: str):
        if backbone_name == "__test_dummy__":
            return None
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            backbone_name,
            trust_remote_code=True,
            use_fast=True,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or tok.unk_token
        return tok

    def encode_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.projector is not None:
            return self.projector(hidden)
        return hidden

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        if not self.backbone_trainable:
            with torch.no_grad():
                hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            hidden = hidden.detach()
        else:
            hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.encode_from_hidden(hidden)

    def encode_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self({"input_ids": input_ids, "attention_mask": attention_mask})

    def trainable_parameters(self):
        params = []
        if self.backbone_trainable:
            params.extend(p for p in self.backbone.parameters() if p.requires_grad)
        if self.projector is not None:
            params.extend(self.projector.parameters())
        return params


def encoder_config_from_contrastive_cfg(cfg) -> EncoderConfig:
    """Construit EncoderConfig depuis ``ContrastiveConfig``."""
    return EncoderConfig(
        backbone_name=str(cfg.backbone_name),
        max_seq_length=int(cfg.max_seq_length),
        backbone_trainable=bool(cfg.backbone_trainable),
        train_last_n_layers=cfg.train_last_n_layers,
        cache_backbone_embeddings=bool(cfg.cache_backbone_embeddings),
        use_projector=bool(cfg.use_projector),
        projection=str(cfg.projection),
        hiddim=int(cfg.hiddim),
    )


def encoder_kwargs_from_cfg(model_cfg: Dict[str, Any]) -> EncoderConfig:
    """Construit EncoderConfig depuis la section YAML ``model`` contrastive."""
    return EncoderConfig(
        backbone_name=str(model_cfg.get("backbone_name")),
        max_seq_length=int(model_cfg.get("max_seq_length", 256)),
        backbone_trainable=bool(model_cfg.get("backbone_trainable", False)),
        train_last_n_layers=model_cfg.get("train_last_n_layers"),
        cache_backbone_embeddings=bool(model_cfg.get("cache_backbone_embeddings", True)),
        use_projector=bool(model_cfg.get("use_projector", True)),
        projection=str(model_cfg.get("projection", "linear")),
        hiddim=int(model_cfg.get("hiddim", 128)),
    )


def build_contrastive_encoder(cfg) -> ContrastiveEncoder:
    return ContrastiveEncoder(encoder_config_from_contrastive_cfg(cfg))


def save_contrastive_encoder_checkpoint(encoder: ContrastiveEncoder, path: str | Path) -> None:
    torch.save(encoder.state_dict(), str(path))


def load_contrastive_encoder(
    cfg: EncoderConfig,
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
) -> ContrastiveEncoder:
    enc = ContrastiveEncoder(cfg)
    try:
        state = torch.load(str(checkpoint_path), map_location=device or "cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(checkpoint_path), map_location=device or "cpu")
    enc.load_state_dict(state)
    if device is not None:
        enc.to(device)
    return enc


def load_contrastive_encoder_from_checkpoint(
    cfg,
    checkpoint_dir: str | Path,
    device: str | torch.device,
) -> ContrastiveEncoder:
    """Charge l'encodeur depuis ``checkpoints/best_model/contrastive_encoder.pt``."""
    checkpoint_dir = Path(checkpoint_dir)
    ckpt_path = checkpoint_dir / CONTRASTIVE_ENCODER_CKPT
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint encodeur absent : {ckpt_path}")
    dev = torch.device(device)
    return load_contrastive_encoder(
        encoder_config_from_contrastive_cfg(cfg),
        ckpt_path,
        device=dev,
    )
