"""Utilitaires d'entraînement HF partagés (encodeur unifié, cache backbone gelé)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.encoder_model import (
    CONTRASTIVE_ENCODER_CKPT,
    ContrastiveEncoder,
    build_contrastive_encoder,
    encoder_config_from_contrastive_cfg,
    load_contrastive_encoder_from_checkpoint,
    save_contrastive_encoder_checkpoint,
)
from contrastive_methods.losses.softtriple import make_collate_fn
from supervised_macro_ft.embedding_cache import should_cache_backbone_embeddings

logger = logging.getLogger(__name__)

METRIC_EVAL_NORMALIZE = True


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_autocast_dtype(device: str, enabled: bool = True) -> Optional[torch.dtype]:
    if not enabled or not str(device).startswith("cuda"):
        return None
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


class TextLabelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str) -> None:
        self.texts = df[text_col].astype(str).tolist()
        self.labels = df["label_id"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"text": self.texts[idx], "label": self.labels[idx]}


class BackboneHiddenDataset(Dataset):
    def __init__(self, hidden: np.ndarray, labels: Sequence[int], indices: Optional[np.ndarray] = None) -> None:
        self.hidden = np.asarray(hidden, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.indices = (
            np.arange(len(labels), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int) -> Dict[str, torch.Tensor]:
        idx = int(self.indices[pos])
        return {
            "hidden": torch.from_numpy(self.hidden[idx]),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


def collate_hidden_batch(items: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "hidden": torch.stack([it["hidden"] for it in items]),
        "labels": torch.stack([it["labels"] for it in items]),
    }


def dataloader_kwargs(device: str) -> Dict[str, Any]:
    if not str(device).startswith("cuda"):
        return {}
    return {"pin_memory": True, "num_workers": 2, "persistent_workers": True}


def model_cfg_dict(cfg: ContrastiveConfig) -> Dict[str, Any]:
    return {
        "backbone_name": cfg.backbone_name,
        "max_seq_length": cfg.max_seq_length,
        "backbone_trainable": cfg.backbone_trainable,
        "train_last_n_layers": cfg.train_last_n_layers,
        "cache_backbone_embeddings": cfg.cache_backbone_embeddings,
        "encode_batch_size": cfg.encode_batch_size,
        "batch_size": cfg.batch_size,
    }


def use_hidden_cache(cfg: ContrastiveConfig) -> bool:
    return should_cache_backbone_embeddings(model_cfg_dict(cfg))


@torch.no_grad()
def encode_backbone_matrix(
    encoder: ContrastiveEncoder,
    texts: Sequence[str],
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if encoder.tokenizer is None:
        raise ValueError("Tokenizer requis pour encoder le backbone.")
    encoder.eval()
    collate_fn = make_collate_fn(encoder.tokenizer, max_length)
    items = [{"text": t, "label": 0} for t in texts]
    loader = DataLoader(items, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    chunks: List[np.ndarray] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        hidden = encoder.backbone(input_ids=input_ids, attention_mask=attention_mask)
        chunks.append(hidden.detach().cpu().numpy().astype(np.float32))
    if not chunks:
        return np.zeros((0, encoder.backbone.hidden_size), dtype=np.float32)
    return np.vstack(chunks)


def load_or_build_backbone_hidden(
    encoder: ContrastiveEncoder,
    texts: Sequence[str],
    cfg: ContrastiveConfig,
    cache_dir: Path,
    device: torch.device,
) -> np.ndarray:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "backbone_hidden.npy"
    n = len(texts)
    if cache_path.is_file():
        arr = np.load(cache_path)
        if arr.ndim == 2 and arr.shape[0] == n:
            logger.info("Cache backbone réutilisé : %s", cache_path)
            return np.asarray(arr, dtype=np.float32)
        logger.warning("Cache backbone ignoré (shape %s, attendu n=%d)", arr.shape, n)
    logger.info("Encodage backbone unique (%d phrases)…", n)
    hidden = encode_backbone_matrix(
        encoder,
        texts,
        max_length=cfg.max_seq_length,
        batch_size=cfg.encode_batch_size,
        device=device,
    )
    if hidden.shape[0] != n:
        raise ValueError(f"Encodage backbone : attendu {n} lignes, obtenu {hidden.shape[0]}")
    np.save(cache_path, hidden)
    return hidden


def build_train_loader(
    cfg: ContrastiveConfig,
    train_df: pd.DataFrame,
    text_col: str,
    encoder: ContrastiveEncoder,
    device: torch.device,
    cache_dir: Path,
    *,
    batch_sampler=None,
) -> Tuple[DataLoader, bool]:
    """Retourne (loader, use_hidden_cache)."""
    dl_kwargs = dataloader_kwargs(str(device))
    use_cache = use_hidden_cache(cfg)
    if use_cache:
        texts = train_df[text_col].astype(str).tolist()
        labels = train_df["label_id"].astype(int).tolist()
        hidden = load_or_build_backbone_hidden(encoder, texts, cfg, cache_dir, device)
        ds = BackboneHiddenDataset(hidden, labels)
        collate = collate_hidden_batch
    else:
        ds = TextLabelDataset(train_df, text_col)
        collate = make_collate_fn(encoder.tokenizer, cfg.max_seq_length)

    if batch_sampler is not None:
        return (
            DataLoader(ds, batch_sampler=batch_sampler, collate_fn=collate, **dl_kwargs),
            use_cache,
        )
    return (
        DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate, **dl_kwargs),
        use_cache,
    )


def build_eval_loader(
    cfg: ContrastiveConfig,
    df: pd.DataFrame,
    text_col: str,
    encoder: ContrastiveEncoder,
    device: torch.device,
) -> DataLoader:
    dl_kwargs = dataloader_kwargs(str(device))
    ds = TextLabelDataset(df, text_col)
    collate = make_collate_fn(encoder.tokenizer, cfg.max_seq_length)
    return DataLoader(
        ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        **dl_kwargs,
    )


def forward_embeddings(
    encoder: ContrastiveEncoder,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    *,
    use_hidden_cache: bool,
) -> torch.Tensor:
    if use_hidden_cache:
        hidden = batch["hidden"].to(device, non_blocking=True)
        return encoder.encode_from_hidden(hidden)
    return encoder(
        {
            "input_ids": batch["input_ids"].to(device, non_blocking=True),
            "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        }
    )


def build_optimizer(
    encoder: ContrastiveEncoder,
    extra_modules: Sequence[torch.nn.Module],
    lr: float,
) -> torch.optim.Optimizer:
    params: List[torch.nn.Parameter] = list(encoder.trainable_parameters())
    for mod in extra_modules:
        params.extend(mod.parameters())
    return torch.optim.AdamW(params, lr=float(lr))


def save_contrastive_checkpoint(
    encoder: ContrastiveEncoder,
    checkpoint_dir: Path,
    *,
    extra_state: Optional[Dict[str, Any]] = None,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_contrastive_encoder_checkpoint(encoder, checkpoint_dir / CONTRASTIVE_ENCODER_CKPT)
    if encoder.tokenizer is not None:
        encoder.tokenizer.save_pretrained(str(checkpoint_dir))
    if extra_state:
        for name, payload in extra_state.items():
            torch.save(payload, checkpoint_dir / name)


def load_contrastive_checkpoint(
    cfg: ContrastiveConfig,
    checkpoint_dir: Path,
    device: str | torch.device,
    *,
    extra_loaders: Optional[Dict[str, Callable[[Any], None]]] = None,
) -> ContrastiveEncoder:
    encoder = load_contrastive_encoder_from_checkpoint(cfg, checkpoint_dir, device)
    if extra_loaders:
        for filename, loader_fn in extra_loaders.items():
            path = Path(checkpoint_dir) / filename
            if path.is_file():
                try:
                    payload = torch.load(path, map_location=device, weights_only=True)
                except TypeError:
                    payload = torch.load(path, map_location=device)
                loader_fn(payload)
    return encoder


@torch.no_grad()
def encode_texts(
    encoder: ContrastiveEncoder,
    texts: Sequence[str],
    cfg: ContrastiveConfig,
    device: str | torch.device,
    *,
    batch_size: Optional[int] = None,
    normalize_embeddings: bool = METRIC_EVAL_NORMALIZE,
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> np.ndarray:
    if encoder.tokenizer is None:
        raise ValueError("Tokenizer requis pour encoder des textes.")
    encoder.eval()
    bs = batch_size or cfg.encode_batch_size
    collate_fn = make_collate_fn(encoder.tokenizer, cfg.max_seq_length)
    items = [{"text": t, "label": 0} for t in texts]
    loader = DataLoader(items, batch_size=bs, shuffle=False, collate_fn=collate_fn)
    dev = torch.device(device)
    autocast_dtype = resolve_autocast_dtype(str(device), cfg.use_amp)
    use_amp = autocast_dtype is not None and dev.type == "cuda"
    parts: List[np.ndarray] = []
    batch_iter: Any = loader
    if show_progress:
        batch_iter = tqdm(
            loader,
            total=len(loader),
            desc=progress_desc or "encode",
            unit="batch",
            file=sys.stdout,
            mininterval=10.0,
            dynamic_ncols=True,
        )
    for batch in batch_iter:
        input_ids = batch["input_ids"].to(dev, non_blocking=True)
        attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
        with torch.autocast(
            device_type=dev.type,
            dtype=autocast_dtype or torch.float32,
            enabled=use_amp,
        ):
            emb = encoder({"input_ids": input_ids, "attention_mask": attention_mask})
        emb_out = emb.detach()
        if normalize_embeddings:
            emb_out = F.normalize(emb_out, p=2, dim=1)
        parts.append(emb_out.float().cpu().numpy())
    if not parts:
        return np.zeros((0, encoder.embedding_dim), dtype=np.float64)
    return np.vstack(parts)


def encode_contrastive_texts(
    cfg: ContrastiveConfig,
    texts: List[str],
    *,
    checkpoint_dir: Optional[Path] = None,
    hf_encoder=None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    normalize_embeddings: bool = METRIC_EVAL_NORMALIZE,
    **_,
) -> np.ndarray:
    """Encode un corpus via l'encodeur HF unifié (checkpoint ou instance fournie)."""
    dev = device or get_device()
    encoder = hf_encoder
    if encoder is None:
        if checkpoint_dir is None:
            raise ValueError("encode contrastif : checkpoint_dir ou hf_encoder requis")
        encoder = load_contrastive_checkpoint(cfg, Path(checkpoint_dir), dev)
    return encode_texts(
        encoder,
        texts,
        cfg,
        dev,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )


def run_training_epoch(
    encoder: ContrastiveEncoder,
    loss_module: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    train: bool,
    use_hidden_cache: bool,
    autocast_dtype: Optional[torch.dtype] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> float:
    encoder.train(mode=train)
    loss_module.train(mode=train)
    use_amp = autocast_dtype is not None and device.type == "cuda"
    total = 0.0
    n_batches = 0
    for batch in loader:
        labels = batch["labels"].to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype or torch.float32,
            enabled=use_amp,
        ):
            emb = forward_embeddings(encoder, batch, device, use_hidden_cache=use_hidden_cache)
            if loss_fn is not None:
                loss = loss_fn(emb, labels)
            else:
                out = loss_module(emb, labels)
                loss = out[0] if isinstance(out, tuple) else out
        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        total += float(loss.detach().float().cpu().item())
        n_batches += 1
    return total / max(1, n_batches)
