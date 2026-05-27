"""TPN full-encoder (end-to-end): mise à jour du backbone via pertes prototypiques."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler

from macro_transfer.constants import LABEL2ID, MACRO_NAMES
from macro_transfer.tpn_eval import evaluate_tpn_transfer
from macro_transfer.tpn_gating import build_gating_frame
from macro_transfer.tpn_prototypes import (
    compute_source_prototypes_torch,
    compute_source_target_prototypes_torch,
    compute_target_prototypes_soft_torch,
    distribution_from_prototypes_torch,
    l2_normalize_np,
    prototype_distance_torch,
    prototype_distance_table,
    prototype_logits_torch,
    soft_assignments_torch,
    symmetric_kl_torch,
)
from scgm_text.checkpoint_io import load_scgm_checkpoint

logger = logging.getLogger(__name__)

EPS = 1e-8


def _to_device(device: str) -> torch.device:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    return torch.device("cpu")


class _TextDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Optional[Sequence[int]] = None,
    ) -> None:
        self.texts = [str(x) for x in texts]
        self.labels = list(labels) if labels is not None else None

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {"text": self.texts[idx], "index": idx}
        if self.labels is not None:
            item["label"] = int(self.labels[idx])
        return item


class _BalancedMacroBatchSampler(Sampler[List[int]]):
    """Batch source équilibré A0/A1/B/C autant que possible."""

    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        drop_last: bool,
        seed: int = 42,
    ) -> None:
        self.labels = np.asarray(labels, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.n_macros = len(MACRO_NAMES)
        self._per_macro = {
            m: np.where(self.labels == m)[0].tolist()
            for m in range(self.n_macros)
        }
        self._len = self._estimate_len()

    def _estimate_len(self) -> int:
        n = len(self.labels)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterable[List[int]]:
        rng = np.random.default_rng(self.seed)
        buckets = {}
        for m, arr in self._per_macro.items():
            arr2 = list(arr)
            rng.shuffle(arr2)
            buckets[m] = arr2
        ptr = {m: 0 for m in range(self.n_macros)}

        macro_order = list(range(self.n_macros))
        for _ in range(self._len):
            batch: List[int] = []
            # Répartition primaire équilibrée
            per_macro_quota = max(1, self.batch_size // self.n_macros)
            for m in macro_order:
                for _q in range(per_macro_quota):
                    if len(batch) >= self.batch_size:
                        break
                    if ptr[m] >= len(buckets[m]):
                        continue
                    batch.append(int(buckets[m][ptr[m]]))
                    ptr[m] += 1
            # Complément avec ce qui reste
            if len(batch) < self.batch_size:
                for m in macro_order:
                    while len(batch) < self.batch_size and ptr[m] < len(buckets[m]):
                        batch.append(int(buckets[m][ptr[m]]))
                        ptr[m] += 1
            if len(batch) < self.batch_size and self.drop_last:
                continue
            if not batch:
                break
            yield batch


@dataclass
class TPNBatch:
    texts: List[str]
    labels: Optional[torch.Tensor] = None


class FullEncoderTPNModel(nn.Module):
    """Backbone texte entraînable pour TPN prototypique end-to-end."""

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

        if self.base_method == "scgm_text":
            if not checkpoint:
                raise ValueError("scgm_text full_encoder requiert --checkpoint")
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
        elif self.base_method == "softtriple":
            from contrastive_methods.losses.softtriple import HFTextEncoder

            if not self.backbone_name:
                self.backbone_name = "Qwen/Qwen3-Embedding-0.6B"
            st_encoder = HFTextEncoder(self.backbone_name, gradient_checkpointing=False)
            if checkpoint:
                ckpt_dir = Path(checkpoint)
                if ckpt_dir.is_file():
                    ckpt_dir = ckpt_dir.parent
                model_bin = ckpt_dir / "hf_model.bin"
                if model_bin.is_file():
                    state = torch.load(model_bin, map_location="cpu")
                    st_encoder.encoder.load_state_dict(state, strict=False)
            self.encoder = st_encoder.encoder
            self.tokenizer = st_encoder.tokenizer
            self.projector = None
            self.kind = "hf_auto"
        else:
            from transformers import AutoModel, AutoTokenizer

            load_ref = checkpoint or self.backbone_name
            if not load_ref:
                raise ValueError("backbone_name ou checkpoint requis pour full_encoder")
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
        # SCGM TextBackbone.forward(input_ids, attention_mask) retourne déjà un tenseur poolé
        # et n'accepte pas return_dict.
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


def _macro_mass(q: torch.Tensor) -> List[float]:
    if q.numel() == 0:
        return [0.0 for _ in MACRO_NAMES]
    return [float(v.item()) for v in q.mean(dim=0)]


def _select_target_by_threshold(
    h_t: torch.Tensor,
    q: torch.Tensor,
    threshold: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    if threshold is None:
        conf = q.max(dim=1).values
        return h_t, q, 1.0, float(conf.mean().item() if len(conf) else 0.0)
    conf, _ = q.max(dim=1)
    mask = conf >= float(threshold)
    if bool(mask.any()):
        h_sel = h_t[mask]
        q_sel = q[mask]
    else:
        # fallback propre: aucune cible confiante -> tensors vides
        h_sel = h_t[:0]
        q_sel = q[:0]
    coverage = float(mask.float().mean().item()) if len(mask) else 0.0
    conf_mean = float(conf.mean().item() if len(conf) else 0.0)
    return h_sel, q_sel, coverage, conf_mean


def _compute_mu_t_with_fallback(
    h_t_sel: torch.Tensor,
    q_sel: torch.Tensor,
    mu_s: torch.Tensor,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    if len(h_t_sel) == 0:
        return mu_s
    mu_t = compute_target_prototypes_soft_torch(h_t_sel, q_sel, eps=eps)
    mass = q_sel.sum(dim=0)
    for m in range(mu_t.shape[0]):
        if float(mass[m].item()) < eps:
            mu_t[m] = mu_s[m]
    return F.normalize(mu_t, p=2, dim=-1, eps=eps)


def compute_tpn_full_encoder_losses(
    model: FullEncoderTPNModel,
    batch_source: TPNBatch,
    batch_target: TPNBatch,
    tpn_cfg: Dict[str, Any],
    loss_weights: Dict[str, float],
    *,
    n_macros: int = 4,
    src_classifier_prototypes: str = "source",
    prototype_mode: str = "batch",
    ema_source_prototypes: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Pertes TPN end-to-end sans conversion numpy dans le chemin différentiable."""
    if batch_source.labels is None:
        raise ValueError("batch_source.labels requis")

    tau = float(tpn_cfg.get("tau", 0.3))
    metric = str(tpn_cfg.get("distance_metric", "euclidean"))
    rho = float(tpn_cfg.get("target_weight_st", 1.0))
    assignment_mode = str(tpn_cfg.get("assignment_mode", "soft"))
    threshold = tpn_cfg.get("pseudo_label_threshold", None)
    threshold_f = float(threshold) if threshold is not None else None

    h_s = model.encode_texts_batch(batch_source.texts)
    h_t = model.encode_texts_batch(batch_target.texts)
    y_ids = batch_source.labels.to(model.device_obj)

    mu_s = compute_source_prototypes_torch(h_s, y_ids, n_macros, eps=EPS)
    if prototype_mode == "ema_global" and ema_source_prototypes is not None:
        mu_s = F.normalize(0.5 * mu_s + 0.5 * ema_source_prototypes.to(mu_s.device), p=2, dim=-1, eps=EPS)

    logits_q = prototype_logits_torch(h_t, mu_s, tau=tau, metric=metric)  # type: ignore[arg-type]
    q = soft_assignments_torch(logits_q, assignment_mode=assignment_mode)  # type: ignore[arg-type]
    h_t_sel, q_sel, pseudo_cov, pseudo_conf_mean = _select_target_by_threshold(h_t, q, threshold_f)

    mu_t = _compute_mu_t_with_fallback(h_t_sel, q_sel, mu_s, eps=EPS)
    if len(h_t_sel) == 0:
        mu_st = mu_s
    else:
        mu_st = compute_source_target_prototypes_torch(
            h_s,
            y_ids,
            h_t_sel,
            q_sel,
            n_macros,
            rho=rho,
            eps=EPS,
        )

    src_proto_mode = str(src_classifier_prototypes).strip().lower()
    src_protos = mu_s if src_proto_mode == "source" else mu_st
    logits_src = prototype_logits_torch(h_s, src_protos, tau=tau, metric=metric)  # type: ignore[arg-type]
    loss_src = F.cross_entropy(logits_src, y_ids)

    proto_terms = []
    for m in range(n_macros):
        proto_terms.append(prototype_distance_torch(mu_s[m], mu_t[m], metric=metric))  # type: ignore[arg-type]
        proto_terms.append(prototype_distance_torch(mu_s[m], mu_st[m], metric=metric))  # type: ignore[arg-type]
        proto_terms.append(prototype_distance_torch(mu_t[m], mu_st[m], metric=metric))  # type: ignore[arg-type]
    loss_proto = torch.stack(proto_terms).mean()

    combined = torch.cat([h_s, h_t], dim=0)
    p_s = distribution_from_prototypes_torch(combined, mu_s, tau=tau, metric=metric)  # type: ignore[arg-type]
    p_t = distribution_from_prototypes_torch(combined, mu_t, tau=tau, metric=metric)  # type: ignore[arg-type]
    p_st = distribution_from_prototypes_torch(combined, mu_st, tau=tau, metric=metric)  # type: ignore[arg-type]
    loss_kl = (
        symmetric_kl_torch(p_s, p_t, eps=EPS)
        + symmetric_kl_torch(p_s, p_st, eps=EPS)
        + symmetric_kl_torch(p_t, p_st, eps=EPS)
    ).mean()

    p_st_t = distribution_from_prototypes_torch(h_t, mu_st, tau=tau, metric=metric)  # type: ignore[arg-type]
    p_clamped = p_st_t.clamp(min=EPS)
    loss_ent = -(p_clamped * torch.log(p_clamped)).sum(dim=1).mean()
    p_bar = p_st_t.mean(dim=0)
    loss_div = (p_bar * torch.log(p_bar + EPS)).sum()
    loss_pres = ((h_s - h_s.detach()) ** 2).sum() * 0.0  # optionnel en full_encoder

    w = loss_weights
    loss_total = (
        float(w.get("src", 1.0)) * loss_src
        + float(w.get("proto", 1.0)) * loss_proto
        + float(w.get("kl", 1.0)) * loss_kl
        + float(w.get("ent", 0.01)) * loss_ent
        + float(w.get("div", 0.01)) * loss_div
        + float(w.get("preserve", 0.0)) * loss_pres
    )
    return {
        "loss_total": loss_total,
        "loss_src": loss_src,
        "loss_proto": loss_proto,
        "loss_kl": loss_kl,
        "loss_ent": loss_ent,
        "loss_div": loss_div,
        "loss_pres": loss_pres,
        "pseudo_coverage": pseudo_cov,
        "pseudo_conf_mean": pseudo_conf_mean,
        "macro_mass_target": _macro_mass(p_st_t.detach()),
        "mu_s": mu_s.detach(),
        "mu_t": mu_t.detach(),
        "mu_st": mu_st.detach(),
    }


def _collate_text_batch(items: List[Dict[str, Any]]) -> TPNBatch:
    texts = [str(x["text"]) for x in items]
    if "label" in items[0]:
        labels = torch.tensor([int(x["label"]) for x in items], dtype=torch.long)
    else:
        labels = None
    return TPNBatch(texts=texts, labels=labels)


def _warmup_lambda(step: int, total_steps: int, warmup_ratio: float) -> float:
    warmup_steps = max(1, int(total_steps * float(warmup_ratio)))
    if step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    return 1.0


@torch.no_grad()
def encode_texts_corpus(
    model: FullEncoderTPNModel,
    texts: Sequence[str],
    *,
    batch_size: int = 32,
) -> np.ndarray:
    model.eval()
    out: List[np.ndarray] = []
    for start in range(0, len(texts), int(batch_size)):
        chunk = texts[start : start + int(batch_size)]
        z = model.encode_texts_batch(chunk)
        out.append(z.detach().cpu().numpy())
    if not out:
        return np.zeros((0, 1), dtype=np.float64)
    return np.asarray(np.vstack(out), dtype=np.float64)


def _compute_global_mu_s(
    z_source: np.ndarray,
    y_source_ids: np.ndarray,
) -> np.ndarray:
    h_s = torch.as_tensor(z_source, dtype=torch.float32)
    y = torch.as_tensor(y_source_ids, dtype=torch.long)
    mu_s = compute_source_prototypes_torch(h_s, y, len(MACRO_NAMES), eps=EPS)
    return np.asarray(mu_s.cpu().numpy(), dtype=np.float64)


def _macro_probs_from_mu_s(
    z_target: np.ndarray,
    mu_s: np.ndarray,
    *,
    tau: float,
    metric: str,
) -> np.ndarray:
    h_t = torch.as_tensor(z_target, dtype=torch.float32)
    mu = torch.as_tensor(mu_s, dtype=torch.float32)
    p = distribution_from_prototypes_torch(h_t, mu, tau=tau, metric=metric)  # type: ignore[arg-type]
    return np.asarray(p.cpu().numpy(), dtype=np.float64)


def _build_metadata_with_probs(meta_target: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    out = meta_target.copy()
    for i, m in enumerate(MACRO_NAMES):
        out[f"p_{m}"] = probs[:, i]
    out["m_hat"] = [MACRO_NAMES[i] for i in probs.argmax(axis=1)]
    p_sorted = np.sort(probs, axis=1)
    out["q_conf"] = p_sorted[:, -1]
    out["margin"] = p_sorted[:, -1] - p_sorted[:, -2]
    out["ambiguous"] = False
    return out


def train_tpn_full_encoder(
    *,
    model: FullEncoderTPNModel,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    out_dir: Path,
    tpn_cfg: Dict[str, Any],
    full_cfg: Dict[str, Any],
    loss_weights: Dict[str, float],
    label_col: str = "pred_label",
    target_label_col: str = "pred_label",
    pred_ok_col_target: str = "pred_ok",
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    emb_dir = out_dir / "embeddings"
    transfer_dir = out_dir / "transfer"
    checkpoints_dir = out_dir / "checkpoints"
    proto_dir = out_dir / "prototypes"
    training_dir = out_dir / "training"
    for p in (emb_dir, transfer_dir, checkpoints_dir, proto_dir, training_dir):
        p.mkdir(parents=True, exist_ok=True)

    src_text_col = str(full_cfg.get("source_text_col", "sentence"))
    tgt_text_col = str(full_cfg.get("target_text_col", "sentence"))
    y_ids = np.array([LABEL2ID[str(x)] for x in source_df[label_col].astype(str)], dtype=np.int64)
    src_dataset = _TextDataset(source_df[src_text_col].astype(str).tolist(), y_ids.tolist())
    tgt_dataset = _TextDataset(target_df[tgt_text_col].astype(str).tolist(), None)

    source_bs = int(full_cfg.get("source_batch_size", 16))
    target_bs = int(full_cfg.get("target_batch_size", 16))
    drop_last = bool(full_cfg.get("drop_last", True))
    balanced = bool(full_cfg.get("balanced_source_batches", True))
    seed = int(full_cfg.get("seed", 42))

    if balanced:
        batch_sampler = _BalancedMacroBatchSampler(y_ids, source_bs, drop_last=drop_last, seed=seed)
        src_loader = DataLoader(src_dataset, batch_sampler=batch_sampler, collate_fn=_collate_text_batch)
    else:
        src_loader = DataLoader(
            src_dataset,
            batch_size=source_bs,
            shuffle=True,
            drop_last=drop_last,
            collate_fn=_collate_text_batch,
        )
    tgt_loader = DataLoader(
        tgt_dataset,
        batch_size=target_bs,
        shuffle=True,
        drop_last=drop_last,
        collate_fn=_collate_text_batch,
    )

    lr = float(full_cfg.get("learning_rate", 2.0e-5))
    wd = float(full_cfg.get("weight_decay", 1.0e-4))
    epochs = int(full_cfg.get("epochs", 5))
    grad_acc = int(full_cfg.get("gradient_accumulation_steps", 1))
    warmup_ratio = float(full_cfg.get("warmup_ratio", 0.1))
    max_grad_norm = float(full_cfg.get("max_grad_norm", 1.0))
    prototype_mode = str(full_cfg.get("prototype_mode", "batch"))
    ema_momentum = float(full_cfg.get("ema_momentum", 0.95))
    src_classifier_prototypes = str(tpn_cfg.get("src_classifier_prototypes", "source"))
    use_fp16 = bool(full_cfg.get("fp16", False))
    use_bf16 = bool(full_cfg.get("bf16", False))
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Aucun paramètre entraînable dans FullEncoderTPNModel.")
    optimizer = AdamW(params, lr=lr, weight_decay=wd)
    total_steps = max(1, epochs * max(1, min(len(src_loader), len(tgt_loader))))
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16 and model.device_obj.type == "cuda")

    best_metric = float("-inf")
    log_rows: List[Dict[str, Any]] = []
    ema_mu_s: Optional[torch.Tensor] = None

    for epoch in range(1, epochs + 1):
        model.train()
        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)
        n_steps = min(len(src_loader), len(tgt_loader))
        if n_steps <= 0:
            raise ValueError("DataLoader vide (source/cible).")
        running = {k: 0.0 for k in ("loss_total", "loss_src", "loss_proto", "loss_kl", "loss_ent", "loss_div")}
        cov_sum = 0.0
        conf_sum = 0.0
        grad_norm_last = 0.0

        optimizer.zero_grad(set_to_none=True)
        for step in range(n_steps):
            src_batch = next(src_iter)
            tgt_batch = next(tgt_iter)
            with torch.autocast(
                device_type=model.device_obj.type,
                dtype=amp_dtype,
                enabled=(model.device_obj.type == "cuda" and (use_fp16 or use_bf16)),
            ):
                losses = compute_tpn_full_encoder_losses(
                    model,
                    src_batch,
                    tgt_batch,
                    tpn_cfg,
                    loss_weights,
                    n_macros=len(MACRO_NAMES),
                    src_classifier_prototypes=src_classifier_prototypes,
                    prototype_mode=prototype_mode,
                    ema_source_prototypes=ema_mu_s,
                )
                loss = losses["loss_total"] / max(1, grad_acc)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % grad_acc == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm_last = float(torch.nn.utils.clip_grad_norm_(params, max_grad_norm).item())
                lmb = _warmup_lambda(step + (epoch - 1) * n_steps, total_steps, warmup_ratio)
                for g in optimizer.param_groups:
                    g["lr"] = lr * lmb
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            for k in running:
                running[k] += float(losses[k].detach().item())
            cov_sum += float(losses["pseudo_coverage"])
            conf_sum += float(losses["pseudo_conf_mean"])
            if prototype_mode == "ema_global":
                cur = losses["mu_s"].detach()
                ema_mu_s = cur if ema_mu_s is None else (ema_momentum * ema_mu_s + (1.0 - ema_momentum) * cur)

        # Evaluation epoch-level (source/target entiers)
        z_source = encode_texts_corpus(model, source_df[src_text_col].astype(str).tolist(), batch_size=source_bs)
        z_target = encode_texts_corpus(model, target_df[tgt_text_col].astype(str).tolist(), batch_size=target_bs)
        z_source = l2_normalize_np(z_source)
        z_target = l2_normalize_np(z_target)
        mu_s_np = _compute_global_mu_s(z_source, y_ids)
        probs = _macro_probs_from_mu_s(
            z_target,
            mu_s_np,
            tau=float(tpn_cfg.get("tau", 0.3)),
            metric=str(tpn_cfg.get("distance_metric", "euclidean")),
        )
        meta_probs = _build_metadata_with_probs(target_df, probs)
        metrics = evaluate_tpn_transfer(
            meta_probs,
            label_col=target_label_col,
            pred_ok_col=pred_ok_col_target,
        )
        macro_f1 = float(metrics.get("macro_f1", float("nan")))
        metric_for_best = macro_f1 if math.isfinite(macro_f1) else float(metrics.get("accuracy", 0.0))

        np.save(emb_dir / "source_full_embeddings.npy", z_source)
        np.save(emb_dir / "target_full_embeddings.npy", z_target)
        # Alias legacy notebooks/exports
        np.save(emb_dir / "source_adapted.npy", z_source)
        np.save(emb_dir / "target_adapted.npy", z_target)
        np.save(emb_dir / "prob_macro_full.npy", probs)
        np.save(emb_dir / "prob_macro_adapted.npy", probs)
        np.savez(proto_dir / "prototypes_full.npz", mu_s=mu_s_np)
        prototype_distance_table(mu_s_np, mu_s_np, mu_s_np).to_csv(
            proto_dir / "prototype_distances_full.csv",
            index=False,
        )
        meta_probs.to_csv(transfer_dir / "metadata_with_tpn_full_macro_probs.csv", index=False)
        meta_probs.to_csv(transfer_dir / "metadata_with_tpn_macro_probs.csv", index=False)
        with open(transfer_dir / "metrics_tpn_full.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        with open(transfer_dir / "transfer_metrics_adapted.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        row = {
            "epoch": epoch,
            "loss_total": running["loss_total"] / n_steps,
            "loss_src": running["loss_src"] / n_steps,
            "loss_proto": running["loss_proto"] / n_steps,
            "loss_kl": running["loss_kl"] / n_steps,
            "loss_ent": running["loss_ent"] / n_steps,
            "loss_div": running["loss_div"] / n_steps,
            "pseudo_coverage": cov_sum / n_steps,
            "pseudo_conf_mean": conf_sum / n_steps,
            "grad_norm": grad_norm_last,
            "macro_f1_eval": float(metrics.get("macro_f1", float("nan"))),
            "balanced_accuracy_eval": float(metrics.get("balanced_accuracy", float("nan"))),
            "accuracy_eval": float(metrics.get("accuracy", float("nan"))),
        }
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(training_dir / "training_log.csv", index=False)
        logger.info(
            "TPN full epoch %d/%d | loss=%.6f macro_f1=%.4f cov=%.3f",
            epoch,
            epochs,
            row["loss_total"],
            row["macro_f1_eval"],
            row["pseudo_coverage"],
        )

        last_state = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "base_method": getattr(model, "base_method", "unknown"),
            "backbone_name": getattr(model, "backbone_name", ""),
            "kind": getattr(model, "kind", "custom"),
            "max_seq_length": getattr(model, "max_seq_length", 256),
            "pooling": getattr(model, "pooling", "mean"),
        }
        (checkpoints_dir / "last_model").mkdir(parents=True, exist_ok=True)
        torch.save(last_state, checkpoints_dir / "last_model" / "model.pt")
        if metric_for_best >= best_metric:
            best_metric = metric_for_best
            (checkpoints_dir / "best_model").mkdir(parents=True, exist_ok=True)
            torch.save(last_state, checkpoints_dir / "best_model" / "model.pt")

    return {
        "output_dir": str(out_dir),
        "best_metric": best_metric,
        "training_log_path": str(training_dir / "training_log.csv"),
        "metrics_path": str(transfer_dir / "metrics_tpn_full.json"),
    }
