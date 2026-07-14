"""Cache embeddings backbone (Qwen gelé) pour entraînement rapide projecteur + tête."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from scgm_text.dataset_text_embeddings import merge_metadata_with_embeddings
from scgm_text.dataset_text_raw import TextRawDataset
from scgm_text.utils_io import create_doc_id_if_missing
from supervised_macro_ft.model import SupervisedMacroModel

logger = logging.getLogger(__name__)


def should_cache_backbone_embeddings(model_cfg: Mapping[str, Any]) -> bool:
    """Active le cache si backbone gelé et option non désactivée."""
    if bool(model_cfg.get("backbone_trainable", True)):
        return False
    return bool(model_cfg.get("cache_backbone_embeddings", True))


def _load_hidden_from_csv(metadata_df: pd.DataFrame, emb_csv: str | Path) -> np.ndarray:
    slim = metadata_df.drop(columns=[c for c in metadata_df.columns if c.startswith("dim_")], errors="ignore")
    meta = create_doc_id_if_missing(slim.copy())
    merged, dim_columns = merge_metadata_with_embeddings(meta, str(emb_csv))
    if len(merged) != len(metadata_df):
        raise ValueError(
            f"Alignement CSV embeddings : metadata={len(metadata_df)}, merged={len(merged)}"
        )
    return merged[dim_columns].to_numpy(dtype=np.float32)


@torch.no_grad()
def encode_backbone_matrix(
    model: SupervisedMacroModel,
    tokenizer,
    texts: Sequence[str],
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> np.ndarray:
    """Encode une seule fois h = pool(Qwen(u)) pour tout le corpus."""
    from scgm_text.collate import make_text_collate_fn
    from supervised_macro_ft.run_logging import batched_progress, log_step_done, log_step_start

    model.eval()
    collate_fn = make_text_collate_fn(tokenizer, max_length)
    items = [{"text": t, "label": 0, "index": i} for i, t in enumerate(texts)]
    loader = DataLoader(items, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    desc = progress_desc or "encode_backbone"
    log_step_start(desc, n_samples=len(texts), batch_size=batch_size, detail="Qwen gelé")
    chunks: list[np.ndarray] = []
    for batch in batched_progress(loader, desc=desc, show_progress=show_progress):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        h = model._encode_backbone(input_ids, attention_mask)
        chunks.append(h.detach().cpu().numpy().astype(np.float32))
    if not chunks:
        return np.zeros((0, 1), dtype=np.float32)
    out = np.vstack(chunks)
    log_step_done(desc, n_samples=len(out))
    return out


def load_backbone_hidden_for_corpus(
    *,
    meta_df: pd.DataFrame,
    texts: Sequence[str],
    emb_csv: str | Path | None,
    cache_path: Path | None,
    model: SupervisedMacroModel,
    tokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Charge h depuis CSV, cache .npy local, ou encode Qwen une fois puis sauvegarde.
    """
    n = len(meta_df)
    if emb_csv:
        emb_path = Path(str(emb_csv))
        if emb_path.is_file():
            logger.info("Chargement embeddings backbone depuis CSV : %s", emb_path)
            return _load_hidden_from_csv(meta_df, emb_path)

    if cache_path is not None and cache_path.is_file():
        arr = np.load(cache_path)
        if arr.ndim == 2 and arr.shape[0] == n:
            logger.info("Cache backbone réutilisé : %s", cache_path)
            return np.asarray(arr, dtype=np.float32)
        logger.warning(
            "Cache backbone ignoré (shape %s, attendu n=%d) : %s",
            arr.shape,
            n,
            cache_path,
        )

    logger.info("Encodage backbone unique (%d phrases) — peut prendre un moment…", n)
    hidden = encode_backbone_matrix(
        model,
        tokenizer,
        texts,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )
    if hidden.shape[0] != n:
        raise ValueError(f"Encodage backbone : attendu {n} lignes, obtenu {hidden.shape[0]}")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, hidden)
        logger.info("Cache backbone sauvegardé : %s", cache_path)
    return hidden


def load_or_build_backbone_hidden(
    *,
    model: SupervisedMacroModel,
    dataset: TextRawDataset,
    tokenizer,
    model_cfg: Mapping[str, Any],
    cache_dir: str | Path,
    device: torch.device,
) -> np.ndarray:
    """
    Charge h depuis CSV, cache .npy, ou encode Qwen une fois puis sauvegarde.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_df = dataset.get_metadata_df()
    max_length = int(model_cfg.get("max_seq_length", 256))
    batch_size = int(model_cfg.get("encode_batch_size", model_cfg.get("batch_size", 32)))
    texts = meta_df[dataset.text_col].astype(str).tolist()
    return load_backbone_hidden_for_corpus(
        meta_df=meta_df,
        texts=texts,
        emb_csv=model_cfg.get("backbone_emb_csv"),
        cache_path=cache_dir / "backbone_hidden.npy",
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )


class BackboneHiddenDataset(Dataset):
    """Sous-ensemble d'embeddings backbone pré-calculés + labels."""

    def __init__(
        self,
        hidden: np.ndarray,
        label_ids: np.ndarray,
        indices: Optional[np.ndarray] = None,
    ) -> None:
        self.hidden = np.asarray(hidden, dtype=np.float32)
        self.label_ids = np.asarray(label_ids, dtype=np.int64)
        self.indices = (
            np.arange(len(label_ids), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[pos])
        return {
            "hidden": torch.from_numpy(self.hidden[idx]),
            "label_ids": torch.tensor(int(self.label_ids[idx]), dtype=torch.long),
        }


def collate_hidden_batch(items: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "hidden": torch.stack([it["hidden"] for it in items]),
        "label_ids": torch.stack([it["label_ids"] for it in items]),
    }


@torch.no_grad()
def encode_projected_matrix(
    model: SupervisedMacroModel,
    hidden: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> np.ndarray:
    """z = ψ(h) par batch (export rapide sans repasser Qwen)."""
    from supervised_macro_ft.run_logging import log_step_done, log_step_start

    model.eval()
    desc = progress_desc or "encode_projected"
    n = len(hidden)
    n_batches = max((n + batch_size - 1) // batch_size, 1)
    log_step_start(desc, n_samples=n, batch_size=batch_size, detail="projecteur ψ")
    chunks: list[np.ndarray] = []
    for batch_idx, start in enumerate(range(0, n, batch_size)):
        if show_progress and (
            batch_idx == 0 or batch_idx == n_batches - 1 or (batch_idx + 1) % 50 == 0
        ):
            logger.info(
                "[macro_ft] %s — batch %d/%d (%.1f%%)",
                desc,
                batch_idx + 1,
                n_batches,
                100.0 * (batch_idx + 1) / n_batches,
            )
        h = torch.from_numpy(hidden[start : start + batch_size]).to(device)
        z = model.encode_from_hidden(h)
        chunks.append(z.detach().cpu().numpy().astype(np.float64))
    if not chunks:
        return np.zeros((0, model.hiddim), dtype=np.float64)
    out = np.vstack(chunks)
    log_step_done(desc, n_samples=len(out))
    return out


@torch.no_grad()
def predict_from_hidden_matrix(
    model: SupervisedMacroModel,
    hidden: np.ndarray,
    *,
    macros: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prédictions macro depuis h pré-calculé (sans Qwen)."""
    model.eval()
    prob_chunks: list[np.ndarray] = []
    n = len(hidden)
    for start in range(0, n, batch_size):
        h = torch.from_numpy(hidden[start : start + batch_size]).to(device)
        logits = model.forward_logits_from_hidden(h)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        prob_chunks.append(probs)
    probs = np.vstack(prob_chunks)
    top = probs.argmax(axis=1)
    pred_macro = np.array([str(macros[i]) for i in top], dtype=object)
    confidence = probs.max(axis=1)
    sort_p = np.sort(probs, axis=1)
    margin = sort_p[:, -1] - sort_p[:, -2] if probs.shape[1] >= 2 else np.zeros(len(probs))
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)
    return pred_macro, probs, confidence, margin, entropy
