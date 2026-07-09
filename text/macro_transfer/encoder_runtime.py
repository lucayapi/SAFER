"""Encodage figé pour Frozen Source Prototypes (backbone + checkpoint)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scgm_text.checkpoint_io import load_scgm_checkpoint

logger = logging.getLogger(__name__)

EPS = 1e-8


def _to_device(device: str) -> torch.device:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    return torch.device("cpu")


class FrozenEncoderModel(nn.Module):
    """Backbone texte gelé pour encodage FSP (sans fine-tuning)."""

    def __init__(
        self,
        *,
        base_method: str,
        checkpoint: Optional[str] = None,
        backbone_name: Optional[str] = None,
        max_seq_length: int = 256,
        pooling: str = "mean",
        freeze_backbone: bool = False,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.base_method = str(base_method)
        self.checkpoint = checkpoint
        self.backbone_name = backbone_name or ""
        self.max_seq_length = int(max_seq_length)
        self.pooling = str(pooling).strip().lower()
        self.freeze_backbone = bool(freeze_backbone)
        self.device_obj = _to_device(device)
        self.kind = "hf_auto"
        self.scgm_model = None
        self.supervised_model = None
        self.contrastive_encoder = None
        t0 = time.monotonic()
        logger.info(
            "FSP encoder init: base_method=%s checkpoint=%s backbone_name=%s freeze_backbone=%s",
            self.base_method,
            checkpoint,
            backbone_name,
            self.freeze_backbone,
        )

        if self.base_method == "scgm_text":
            if not checkpoint:
                raise ValueError("scgm_text requiert un checkpoint")
            model, ckpt_args, _raw = load_scgm_checkpoint(checkpoint, map_location="cpu")
            self.scgm_model = model
            self.encoder = model.backbone
            self.projector = model.projector
            self.backbone_name = str(
                ckpt_args.get("backbone_model_name_or_path")
                or ckpt_args.get("backbone_name")
                or self.backbone_name
            )
            self.kind = "scgm"
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.backbone_name,
                trust_remote_code=True,
            )
        elif self.base_method in ("softtriple", "softtriple_native", "supcon", "batch_triplet"):
            from contrastive_methods.config import ContrastiveConfig
            from contrastive_methods.encoder_model import load_contrastive_encoder_from_checkpoint

            if not checkpoint:
                raise ValueError(f"{self.base_method} requiert un checkpoint contrastif")
            ckpt_dir = Path(checkpoint)
            if ckpt_dir.is_file():
                ckpt_dir = ckpt_dir.parent
            method_name = (
                "softtriple"
                if self.base_method in ("softtriple", "softtriple_native")
                else self.base_method
            )
            if not self.backbone_name:
                self.backbone_name = "Qwen/Qwen3-Embedding-0.6B"
            cfg = ContrastiveConfig(
                method_name=method_name,
                dataset_path=Path("."),
                backbone_name=self.backbone_name,
                max_seq_length=self.max_seq_length,
            )
            self.contrastive_encoder = load_contrastive_encoder_from_checkpoint(
                cfg,
                ckpt_dir,
                str(self.device_obj),
            )
            self.tokenizer = self.contrastive_encoder.tokenizer
            self.encoder = None
            self.projector = None
            self.kind = "contrastive"
        elif self.base_method == "supervised_macro_ft":
            from supervised_macro_ft.checkpoint_io import load_checkpoint, read_checkpoint_config

            if not checkpoint:
                raise ValueError("supervised_macro_ft requiert un checkpoint")
            ckpt_dir = Path(checkpoint)
            ckpt_cfg = read_checkpoint_config(ckpt_dir)
            self.backbone_name = str(ckpt_cfg.get("backbone_name") or self.backbone_name or "Qwen/Qwen3-Embedding-0.6B")
            self.supervised_model = load_checkpoint(ckpt_dir, device=str(self.device_obj))
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.backbone_name, trust_remote_code=True, use_fast=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
            self.encoder = self.supervised_model.backbone.encoder
            self.projector = None
            self.kind = "supervised_macro_ft"
        else:
            from transformers import AutoModel, AutoTokenizer

            load_ref = checkpoint or self.backbone_name
            if not load_ref:
                raise ValueError("backbone_name ou checkpoint requis pour l'encodeur")
            self.tokenizer = AutoTokenizer.from_pretrained(
                load_ref,
                trust_remote_code=True,
                use_fast=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
            self.encoder = AutoModel.from_pretrained(
                load_ref,
                trust_remote_code=True,
            )
            self.projector = None
            self.kind = "hf_auto"

        if self.freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
            if self.projector is not None:
                for p in self.projector.parameters():
                    p.requires_grad = False
        self.to(self.device_obj)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "FSP encoder ready: kind=%s device=%s params_trainable=%d/%d elapsed=%.1fs",
            self.kind,
            self.device_obj,
            trainable_params,
            all_params,
            time.monotonic() - t0,
        )

    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

    def encode_texts_batch(self, texts: Sequence[str]) -> torch.Tensor:
        enc = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device_obj) for k, v in enc.items()}
        if self.kind == "supervised_macro_ft":
            return self.supervised_model.encode(enc["input_ids"], enc["attention_mask"])
        if self.kind == "contrastive":
            h = self.contrastive_encoder(
                {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
            )
            return F.normalize(h, p=2, dim=-1, eps=EPS)
        if self.kind == "scgm":
            h = self.encoder(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
        else:
            outputs = self.encoder(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                return_dict=True,
            )
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                h = self._mean_pool(outputs.last_hidden_state, enc["attention_mask"])
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                h = outputs.pooler_output
            else:
                raise ValueError("Backbone ne retourne ni last_hidden_state ni pooler_output.")
        if self.projector is not None:
            h = self.projector(h)
        return F.normalize(h, p=2, dim=-1, eps=EPS)


@torch.no_grad()
def encode_texts_corpus(
    model: FrozenEncoderModel,
    texts: Sequence[str],
    *,
    batch_size: int = 32,
    log_label: str = "corpus",
) -> np.ndarray:
    model.eval()
    total = len(texts)
    n_batches = (total + int(batch_size) - 1) // int(batch_size) if total else 0
    t0 = time.monotonic()
    logger.info(
        "FSP encode start [%s]: n_texts=%d batch_size=%d n_batches=%d",
        log_label,
        total,
        int(batch_size),
        n_batches,
    )
    out: List[np.ndarray] = []
    for bi, start in enumerate(range(0, len(texts), int(batch_size)), start=1):
        chunk = texts[start : start + int(batch_size)]
        z = model.encode_texts_batch(chunk)
        out.append(z.detach().cpu().numpy())
        done = min(start + int(batch_size), total)
        if bi == 1 or bi == n_batches or bi % max(1, n_batches // 10 or 1) == 0:
            elapsed = max(1e-6, time.monotonic() - t0)
            rate = done / elapsed
            eta = (total - done) / rate if rate > 0 else 0.0
            logger.info(
                "FSP encode [%s] batch %d/%d | %d/%d (%.1f%%) elapsed=%.0fs eta=%.0fs",
                log_label,
                bi,
                n_batches,
                done,
                total,
                (100.0 * done / total) if total else 100.0,
                elapsed,
                eta,
            )
    if not out:
        return np.zeros((0, 1), dtype=np.float64)
    arr = np.asarray(np.vstack(out), dtype=np.float64)
    logger.info(
        "FSP encode done [%s]: shape=%s elapsed=%.1fs",
        log_label,
        tuple(arr.shape),
        time.monotonic() - t0,
    )
    return arr
