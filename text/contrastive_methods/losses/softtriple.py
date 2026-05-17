"""SoftTriple loss + encodeur HF (mean pooling)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class HFTextEncoder(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        hf_cache_folder: Optional[str] = None,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        config = AutoConfig.from_pretrained(
            base_model_name,
            cache_dir=hf_cache_folder,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            cache_dir=hf_cache_folder,
            trust_remote_code=True,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        self.encoder = AutoModel.from_pretrained(
            base_model_name,
            cache_dir=hf_cache_folder,
            trust_remote_code=True,
            config=config,
        )
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        hidden_size = getattr(config, "hidden_size", None) or getattr(config, "d_model", None)
        if hidden_size is None:
            raise ValueError("Impossible d'inférer hidden_size depuis le config du modèle.")
        self.embedding_dim = int(hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return mean_pooling(outputs.last_hidden_state, attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        raise ValueError("Le modèle HF ne retourne ni last_hidden_state ni pooler_output.")


class SoftTripleLoss(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        centers_per_class: int = 5,
        gamma: float = 0.1,
        la: float = 10.0,
        delta: float = 0.01,
        tau: float = 0.0,
        normalize_embeddings: bool = True,
        normalize_centers: bool = True,
        center_max_similarity: float = 0.50,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.centers_per_class = int(centers_per_class)
        self.gamma = float(gamma)
        self.la = float(la)
        self.delta = float(delta)
        self.tau = float(tau)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.normalize_centers = bool(normalize_centers)
        self.center_max_similarity = float(center_max_similarity)
        centers = torch.randn(num_classes, centers_per_class, embedding_dim) * 0.02
        centers = F.normalize(centers, p=2, dim=-1)
        self.centers = nn.Parameter(centers)

    def _get_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.normalize_embeddings:
            return F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def _get_centers(self) -> torch.Tensor:
        if self.normalize_centers:
            return F.normalize(self.centers, p=2, dim=-1)
        return self.centers

    def compute_relaxed_class_similarity(
        self, embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self._get_embeddings(embeddings)
        centers = self._get_centers()
        raw_sim = torch.einsum("bd,ckd->bck", z, centers)
        if self.centers_per_class == 1:
            relaxed_sim = raw_sim.squeeze(-1)
        else:
            q = F.softmax(raw_sim / max(self.gamma, 1e-8), dim=2)
            relaxed_sim = (q * raw_sim).sum(dim=2)
        return relaxed_sim, raw_sim

    def regularization(self) -> torch.Tensor:
        if self.tau <= 0.0 or self.centers_per_class <= 1:
            return torch.tensor(0.0, device=self.centers.device)
        centers = self._get_centers()
        penalties = []
        iu = torch.triu_indices(
            self.centers_per_class, self.centers_per_class, offset=1, device=centers.device
        )
        for c in range(self.num_classes):
            wc = centers[c]
            sim = torch.clamp(wc @ wc.T, -1.0, 1.0)
            pair_sim = sim[iu[0], iu[1]]
            penalty = F.relu(pair_sim - self.center_max_similarity).pow(2)
            if penalty.numel() > 0:
                penalties.append(penalty.mean())
        if not penalties:
            return torch.tensor(0.0, device=self.centers.device)
        return self.tau * torch.stack(penalties).mean()

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        relaxed_sim, _ = self.compute_relaxed_class_similarity(embeddings)
        logits = self.la * relaxed_sim
        batch_idx = torch.arange(labels.shape[0], device=labels.device)
        logits = logits.clone()
        logits[batch_idx, labels] = self.la * (
            relaxed_sim[batch_idx, labels] - self.delta
        )
        ce = F.cross_entropy(logits, labels)
        reg = self.regularization()
        loss = ce + reg
        return loss, {
            "loss_total": float(loss.detach().cpu().item()),
            "loss_ce": float(ce.detach().cpu().item()),
            "loss_reg": float(reg.detach().cpu().item()),
        }


class _TextBatchDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"text": self.texts[idx], "label": self.labels[idx]}


def make_collate_fn(tokenizer, max_length: int):
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [x["text"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

    return collate


@torch.no_grad()
def encode_texts_with_hf_encoder(
    model: HFTextEncoder,
    texts: List[str],
    *,
    batch_size: int,
    device: str,
    normalize_embeddings: bool = True,
    max_length: Optional[int] = None,
) -> np.ndarray:
    model.eval()
    if max_length is None:
        max_length = min(int(getattr(model.tokenizer, "model_max_length", 512) or 512), 512)
    collate_fn = make_collate_fn(model.tokenizer, max_length)
    ds = _TextBatchDataset(texts, [0] * len(texts))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    parts = []
    dev = torch.device(device)
    for batch in dl:
        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch["attention_mask"].to(dev)
        emb = model(input_ids, attention_mask)
        if normalize_embeddings:
            emb = F.normalize(emb, p=2, dim=1)
        parts.append(emb.cpu().numpy())
    return np.vstack(parts) if parts else np.zeros((0, model.embedding_dim), dtype=np.float32)
