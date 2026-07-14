"""Inférence encodeur + prédiction corpus (supervised_macro_ft)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from scgm_text.collate import make_text_collate_fn


@torch.no_grad()
def encode_texts(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> np.ndarray:
    from supervised_macro_ft.run_logging import batched_progress, log_step_done, log_step_start

    model.eval()
    collate_fn = make_text_collate_fn(tokenizer, max_length)
    dummy_labels = [{"text": t, "label": 0, "index": i} for i, t in enumerate(texts)]
    loader = DataLoader(dummy_labels, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    desc = progress_desc or "encode_z"
    log_step_start(desc, n_samples=len(texts), batch_size=batch_size, detail="Qwen + projecteur")
    chunks: list[np.ndarray] = []
    for batch in batched_progress(loader, desc=desc, show_progress=show_progress):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        h = model.encode(input_ids, attention_mask)
        chunks.append(h.detach().cpu().numpy())
    if not chunks:
        return np.zeros((0, 1), dtype=np.float64)
    out = np.vstack(chunks).astype(np.float64)
    log_step_done(desc, n_samples=len(out))
    return out


@torch.no_grad()
def predict_corpus(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    macros: Sequence[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    collate_fn = make_text_collate_fn(tokenizer, max_length)
    items = [{"text": t, "label": 0, "index": i} for i, t in enumerate(texts)]
    loader = DataLoader(items, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    prob_chunks: list[np.ndarray] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        probs, _ = model.predict_proba(input_ids, attention_mask)
        prob_chunks.append(probs.detach().cpu().numpy())
    probs = np.vstack(prob_chunks)
    top = probs.argmax(axis=1)
    pred_macro = np.array([str(macros[i]) for i in top], dtype=object)
    confidence = probs.max(axis=1)
    sort_p = np.sort(probs, axis=1)
    margin = sort_p[:, -1] - sort_p[:, -2] if probs.shape[1] >= 2 else np.zeros(len(probs))
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)
    return pred_macro, probs, confidence, margin, entropy
