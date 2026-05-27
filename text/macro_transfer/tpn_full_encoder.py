"""TPN full-encoder (end-to-end): mise à jour du backbone via pertes prototypiques."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

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


def _macro_batch_quotas(batch_size: int, n_macros: int, min_per_macro: int) -> List[int]:
    """Répartition stable du batch entre macros (reste distribué par rotation)."""
    if batch_size <= 0:
        raise ValueError(f"batch_size doit être > 0, reçu {batch_size}")
    if min_per_macro * n_macros > batch_size:
        raise ValueError(
            f"min_per_macro={min_per_macro} * n_macros={n_macros} "
            f"excède batch_size={batch_size}"
        )
    quotas = [int(min_per_macro)] * n_macros
    remaining = batch_size - sum(quotas)
    idx = 0
    while remaining > 0:
        quotas[idx % n_macros] += 1
        remaining -= 1
        idx += 1
    return quotas


def _source_macro_counts(labels: Sequence[int]) -> Dict[str, int]:
    arr = np.asarray(labels, dtype=np.int64)
    return {name: int((arr == mid).sum()) for mid, name in enumerate(MACRO_NAMES)}


def _validate_source_macros_present(labels: Sequence[int], *, n_macros: int = 4) -> None:
    missing = [MACRO_NAMES[m] for m in range(n_macros) if not np.any(np.asarray(labels) == m)]
    if missing:
        raise ValueError(
            "Macros absentes du dataset source filtré: "
            f"{missing}. Impossible d'entraîner TPN full encoder sans toutes les macros."
        )


class _BalancedMacroBatchSampler(Sampler[List[int]]):
    """Batch source équilibré A0/A1/B/C avec remplacement optionnel."""

    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        drop_last: bool,
        seed: int = 42,
        *,
        min_per_macro: int = 1,
        with_replacement: bool = True,
        n_macros: int = 4,
    ) -> None:
        self.labels = np.asarray(labels, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.min_per_macro = int(min_per_macro)
        self.with_replacement = bool(with_replacement)
        self.n_macros = int(n_macros)
        _validate_source_macros_present(self.labels, n_macros=self.n_macros)
        self._per_macro = {
            m: np.where(self.labels == m)[0].tolist()
            for m in range(self.n_macros)
        }
        self.macro_counts = _source_macro_counts(self.labels)
        self._len = self._estimate_len()

    def _estimate_len(self) -> int:
        n = len(self.labels)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __len__(self) -> int:
        return self._len

    def _draw_from_macro(
        self,
        rng: np.random.Generator,
        macro_id: int,
        count: int,
        buckets: Dict[int, List[int]],
        ptr: Dict[int, int],
    ) -> List[int]:
        if count <= 0:
            return []
        bucket = buckets[macro_id]
        if self.with_replacement:
            if not bucket:
                return []
            chosen = rng.choice(bucket, size=count, replace=True)
            return [int(x) for x in chosen.tolist()]
        picked: List[int] = []
        while len(picked) < count and ptr[macro_id] < len(bucket):
            picked.append(int(bucket[ptr[macro_id]]))
            ptr[macro_id] += 1
        return picked

    def __iter__(self) -> Iterable[List[int]]:
        rng = np.random.default_rng(self.seed)
        buckets = {}
        for m, arr in self._per_macro.items():
            arr2 = list(arr)
            rng.shuffle(arr2)
            buckets[m] = arr2
        ptr = {m: 0 for m in range(self.n_macros)}
        quotas_template = _macro_batch_quotas(
            self.batch_size,
            self.n_macros,
            self.min_per_macro,
        )

        for _ in range(self._len):
            batch: List[int] = []
            for m, quota in enumerate(quotas_template):
                batch.extend(self._draw_from_macro(rng, m, quota, buckets, ptr))
            if len(batch) < self.batch_size and not self.with_replacement:
                for m in range(self.n_macros):
                    while len(batch) < self.batch_size and ptr[m] < len(buckets[m]):
                        batch.append(int(buckets[m][ptr[m]]))
                        ptr[m] += 1
            if len(batch) < self.batch_size and self.drop_last:
                continue
            if not batch:
                break
            rng.shuffle(batch)
            yield batch[: self.batch_size]


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
        t0 = time.monotonic()
        logger.info(
            "TPN full model init: base_method=%s checkpoint=%s backbone_name=%s freeze_backbone=%s",
            self.base_method,
            checkpoint,
            backbone_name,
            self.freeze_backbone,
        )

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
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "TPN full model ready: kind=%s device=%s params_trainable=%d/%d elapsed=%.1fs",
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


def _parse_threshold_per_macro(
    threshold_cfg: Optional[Dict[str, Any]],
    n_macros: int,
    default_threshold: float,
) -> List[float]:
    thresholds = [float(default_threshold)] * n_macros
    if not threshold_cfg:
        return thresholds
    for key, val in threshold_cfg.items():
        key_s = str(key)
        if key_s in LABEL2ID:
            thresholds[LABEL2ID[key_s]] = float(val)
        elif key_s.isdigit():
            idx = int(key_s)
            if 0 <= idx < n_macros:
                thresholds[idx] = float(val)
    return thresholds


def build_target_pseudo_mask(
    q: torch.Tensor,
    *,
    strategy: str = "global_threshold",
    assignment_mode: str = "soft",
    global_threshold: Optional[float] = None,
    threshold_per_macro: Optional[Sequence[float]] = None,
    min_per_macro: int = 8,
    min_confidence: float = 0.25,
) -> torch.Tensor:
    """
    Retourne un masque M [N_t, n_class] indiquant quelles paires (exemple, macro)
    contribuent aux prototypes cible.
    """
    if q.ndim != 2:
        raise ValueError(f"q doit être 2D, reçu shape={tuple(q.shape)}")
    n_t, n_class = q.shape
    device = q.device
    dtype = q.dtype
    strategy_norm = str(strategy).strip().lower()
    mode = str(assignment_mode).strip().lower()
    mask = torch.zeros((n_t, n_class), device=device, dtype=dtype)

    if strategy_norm == "global_threshold":
        conf, top = q.max(dim=1)
        if global_threshold is None:
            keep = torch.ones(n_t, device=device, dtype=torch.bool)
        else:
            keep = conf >= float(global_threshold)
        if mode == "hard":
            rows = torch.arange(n_t, device=device)
            mask[rows[keep], top[keep]] = 1.0
        else:
            mask = q * keep.to(dtype).unsqueeze(1)
        return mask

    per_macro_thr = list(threshold_per_macro or [global_threshold or 0.6] * n_class)

    if strategy_norm == "per_class_threshold":
        if mode == "hard":
            top = q.argmax(dim=1)
            rows = torch.arange(n_t, device=device)
            for j in range(n_t):
                m_star = int(top[j].item())
                if float(q[j, m_star].item()) >= float(per_macro_thr[m_star]):
                    mask[j, m_star] = 1.0
        else:
            thr = torch.as_tensor(per_macro_thr, device=device, dtype=dtype).view(1, -1)
            mask = (q >= thr).to(dtype)
        return mask

    if strategy_norm == "per_class_topk":
        k_default = max(1, int(min_per_macro))
        min_conf = float(min_confidence)
        selected_any = torch.zeros(n_t, device=device, dtype=torch.bool)
        per_macro_selected: List[torch.Tensor] = []
        for m in range(n_class):
            k = min(k_default, n_t)
            if k <= 0:
                per_macro_selected.append(torch.zeros(0, dtype=torch.long, device=device))
                continue
            vals, idx = torch.topk(q[:, m], k=k, largest=True)
            ok = vals >= min_conf
            idx_ok = idx[ok]
            per_macro_selected.append(idx_ok)
            if idx_ok.numel():
                selected_any[idx_ok] = True

        if mode == "hard":
            top = q.argmax(dim=1)
            rows = torch.arange(n_t, device=device)
            for m in range(n_class):
                idx_ok = per_macro_selected[m]
                if idx_ok.numel() == 0:
                    continue
                same_top = top[idx_ok] == m
                chosen = idx_ok[same_top]
                if chosen.numel():
                    mask[chosen, m] = 1.0
            # Si aucun hard match, fallback: top-1 parmi les exemples sélectionnés
            if mask.sum() == 0 and bool(selected_any.any()):
                sel_rows = rows[selected_any]
                mask[sel_rows, top[selected_any]] = 1.0
        else:
            for m in range(n_class):
                idx_ok = per_macro_selected[m]
                if idx_ok.numel():
                    mask[idx_ok, m] = 1.0
        return mask

    raise ValueError(f"pseudo_label_strategy inconnu: {strategy}")


def _macro_mass_from_q_masked(q_masked: torch.Tensor) -> List[float]:
    if q_masked.numel() == 0:
        return [0.0 for _ in MACRO_NAMES]
    return [float(v.item()) for v in q_masked.sum(dim=0)]


def _pseudo_coverage_by_macro(mask: torch.Tensor) -> List[float]:
    if mask.numel() == 0:
        return [0.0 for _ in MACRO_NAMES]
    n_t = mask.shape[0]
    return [float((mask[:, m] > 0).float().mean().item()) for m in range(mask.shape[1])]


def _pseudo_conf_mean_by_macro(q: torch.Tensor, mask: torch.Tensor) -> List[float]:
    out: List[float] = []
    for m in range(q.shape[1]):
        active = mask[:, m] > 0
        if not bool(active.any()):
            out.append(0.0)
            continue
        out.append(float(q[active, m].mean().item()))
    return out


def _prototype_validity_source(y_ids: torch.Tensor, n_macros: int, *, eps: float = EPS) -> List[bool]:
    valid: List[bool] = []
    for m in range(n_macros):
        valid.append(bool((y_ids == m).any().item()))
    return valid


def _prototype_validity_target(q_masked: torch.Tensor, *, eps: float = EPS) -> List[bool]:
    if q_masked.numel() == 0:
        return [False for _ in MACRO_NAMES]
    mass = q_masked.sum(dim=0)
    return [float(mass[m].item()) >= eps for m in range(q_masked.shape[1])]


def _format_macro_metrics(prefix: str, values: Sequence[float]) -> str:
    parts = [f"{MACRO_NAMES[i]}={values[i]:.4f}" for i in range(min(len(values), len(MACRO_NAMES)))]
    return f"{prefix}: " + " ".join(parts)


def _format_macro_bool(prefix: str, values: Sequence[bool]) -> str:
    parts = [f"{MACRO_NAMES[i]}={'True' if values[i] else 'False'}" for i in range(min(len(values), len(MACRO_NAMES)))]
    return f"{prefix}: " + " ".join(parts)


def _source_batch_counts(y_ids: torch.Tensor) -> List[int]:
    return [int((y_ids == m).sum().item()) for m in range(len(MACRO_NAMES))]


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
    pseudo_strategy = str(tpn_cfg.get("pseudo_label_strategy", "global_threshold"))
    global_threshold = tpn_cfg.get("pseudo_label_threshold", None)
    global_threshold_f = float(global_threshold) if global_threshold is not None else None
    min_confidence = float(tpn_cfg.get("pseudo_label_min_confidence", 0.25))
    min_per_macro_pseudo = int(tpn_cfg.get("pseudo_label_min_per_macro", 8))
    threshold_per_macro = _parse_threshold_per_macro(
        tpn_cfg.get("pseudo_label_threshold_per_macro"),
        n_macros,
        global_threshold_f if global_threshold_f is not None else 0.6,
    )

    h_s = model.encode_texts_batch(batch_source.texts)
    h_t = model.encode_texts_batch(batch_target.texts)
    y_ids = batch_source.labels.to(model.device_obj)

    mu_s = compute_source_prototypes_torch(h_s, y_ids, n_macros, eps=EPS)
    if prototype_mode == "ema_global" and ema_source_prototypes is not None:
        mu_s = F.normalize(0.5 * mu_s + 0.5 * ema_source_prototypes.to(mu_s.device), p=2, dim=-1, eps=EPS)

    logits_q = prototype_logits_torch(h_t, mu_s, tau=tau, metric=metric)  # type: ignore[arg-type]
    q = soft_assignments_torch(logits_q, assignment_mode=assignment_mode)  # type: ignore[arg-type]
    pseudo_mask = build_target_pseudo_mask(
        q,
        strategy=pseudo_strategy,
        assignment_mode=assignment_mode,
        global_threshold=global_threshold_f,
        threshold_per_macro=threshold_per_macro,
        min_per_macro=min_per_macro_pseudo,
        min_confidence=min_confidence,
    )
    q_masked = q * pseudo_mask
    pseudo_cov_by_macro = _pseudo_coverage_by_macro(pseudo_mask)
    pseudo_cov = float(np.mean(pseudo_cov_by_macro)) if pseudo_cov_by_macro else 0.0
    pseudo_conf_by_macro = _pseudo_conf_mean_by_macro(q, pseudo_mask)
    pseudo_conf_mean = float(np.mean(pseudo_conf_by_macro)) if pseudo_conf_by_macro else 0.0
    target_mass_by_macro = _macro_mass_from_q_masked(q_masked)
    source_counts = _source_batch_counts(y_ids)
    source_proto_valid = _prototype_validity_source(y_ids, n_macros, eps=EPS)
    target_proto_valid = _prototype_validity_target(q_masked, eps=EPS)

    mu_t = compute_target_prototypes_soft_torch(h_t, q_masked, eps=EPS)
    mass_t = q_masked.sum(dim=0)
    for m in range(mu_t.shape[0]):
        if float(mass_t[m].item()) < EPS:
            mu_t[m] = mu_s[m]
    mu_t = F.normalize(mu_t, p=2, dim=-1, eps=EPS)

    if float(q_masked.sum().item()) <= EPS:
        mu_st = mu_s
    else:
        mu_st = compute_source_target_prototypes_torch(
            h_s,
            y_ids,
            h_t,
            q_masked,
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
    loss_reg = ((h_s - h_s.detach()) ** 2).sum() * 0.0  # placeholder optionnel

    w = loss_weights
    w_reg = float(w.get("reg", w.get("preserve", 0.0)))
    loss_total = (
        float(w.get("src", 1.0)) * loss_src
        + float(w.get("proto", 1.0)) * loss_proto
        + float(w.get("kl", 1.0)) * loss_kl
        + float(w.get("ent", 0.01)) * loss_ent
        + float(w.get("div", 0.01)) * loss_div
        + w_reg * loss_reg
    )
    return {
        "loss_total": loss_total,
        "loss_src": loss_src,
        "loss_proto": loss_proto,
        "loss_kl": loss_kl,
        "loss_ent": loss_ent,
        "loss_div": loss_div,
        "loss_reg": loss_reg,
        "loss_pres": loss_reg,
        "pseudo_coverage": pseudo_cov,
        "pseudo_conf_mean": pseudo_conf_mean,
        "macro_mass_target": target_mass_by_macro,
        "source_batch_counts": source_counts,
        "target_mass_by_macro": target_mass_by_macro,
        "pseudo_coverage_by_macro": pseudo_cov_by_macro,
        "pseudo_conf_mean_by_macro": pseudo_conf_by_macro,
        "source_proto_valid": source_proto_valid,
        "target_proto_valid": target_proto_valid,
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
    log_label: str = "corpus",
) -> np.ndarray:
    model.eval()
    total = len(texts)
    n_batches = (total + int(batch_size) - 1) // int(batch_size) if total else 0
    t0 = time.monotonic()
    logger.info(
        "TPN encode start [%s]: n_texts=%d batch_size=%d n_batches=%d",
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
                "TPN encode [%s] batch %d/%d | %d/%d (%.1f%%) elapsed=%.0fs eta=%.0fs",
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
        "TPN encode done [%s]: shape=%s elapsed=%.1fs",
        log_label,
        tuple(arr.shape),
        time.monotonic() - t0,
    )
    return arr


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
    with_replacement = bool(full_cfg.get("source_batch_with_replacement", True))
    min_per_macro = int(full_cfg.get("min_per_macro", 1))
    seed = int(full_cfg.get("seed", 42))
    progress_every = int(full_cfg.get("progress_every", 25))

    _validate_source_macros_present(y_ids, n_macros=len(MACRO_NAMES))
    source_counts_global = _source_macro_counts(y_ids)
    logger.info("Source distribution (filtered): %s", source_counts_global)

    if balanced:
        batch_sampler = _BalancedMacroBatchSampler(
            y_ids,
            source_bs,
            drop_last=drop_last,
            seed=seed,
            min_per_macro=min_per_macro,
            with_replacement=with_replacement,
            n_macros=len(MACRO_NAMES),
        )
        logger.info(
            "Balanced source sampler: batch_size=%d min_per_macro=%d with_replacement=%s "
            "estimated_batches=%d drop_last=%s",
            source_bs,
            min_per_macro,
            with_replacement,
            len(batch_sampler),
            drop_last,
        )
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
    warned_missing_source_macro = False

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
        diag_source_counts = np.zeros(len(MACRO_NAMES), dtype=np.float64)
        diag_target_mass = np.zeros(len(MACRO_NAMES), dtype=np.float64)
        diag_pseudo_cov = np.zeros(len(MACRO_NAMES), dtype=np.float64)
        diag_pseudo_conf = np.zeros(len(MACRO_NAMES), dtype=np.float64)
        diag_steps = 0

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

            src_counts = losses.get("source_batch_counts", [])
            if src_counts:
                diag_source_counts += np.asarray(src_counts, dtype=np.float64)
            diag_target_mass += np.asarray(losses.get("target_mass_by_macro", [0.0] * len(MACRO_NAMES)), dtype=np.float64)
            diag_pseudo_cov += np.asarray(losses.get("pseudo_coverage_by_macro", [0.0] * len(MACRO_NAMES)), dtype=np.float64)
            diag_pseudo_conf += np.asarray(
                losses.get("pseudo_conf_mean_by_macro", [0.0] * len(MACRO_NAMES)),
                dtype=np.float64,
            )
            diag_steps += 1

            if balanced and with_replacement and src_counts:
                missing_in_batch = [
                    MACRO_NAMES[i]
                    for i, c in enumerate(src_counts)
                    if int(c) < min_per_macro
                ]
                if missing_in_batch and not warned_missing_source_macro:
                    logger.warning(
                        "Batch source sans min_per_macro=%d pour macros %s malgré with_replacement=true "
                        "(epoch=%d step=%d counts=%s)",
                        min_per_macro,
                        missing_in_batch,
                        epoch,
                        step + 1,
                        {MACRO_NAMES[i]: int(src_counts[i]) for i in range(len(src_counts))},
                    )
                    warned_missing_source_macro = True

            if progress_every > 0 and ((step + 1) % progress_every == 0 or step == 0 or step + 1 == n_steps):
                logger.info(
                    "TPN full diag epoch=%d step=%d/%d | %s | %s | %s | %s | %s | %s",
                    epoch,
                    step + 1,
                    n_steps,
                    _format_macro_metrics(
                        "source_batch_counts",
                        (diag_source_counts / max(1, diag_steps)).tolist(),
                    ),
                    _format_macro_metrics(
                        "target_mass_by_macro",
                        (diag_target_mass / max(1, diag_steps)).tolist(),
                    ),
                    _format_macro_metrics(
                        "pseudo_coverage_by_macro",
                        (diag_pseudo_cov / max(1, diag_steps)).tolist(),
                    ),
                    _format_macro_metrics(
                        "pseudo_conf_mean_by_macro",
                        (diag_pseudo_conf / max(1, diag_steps)).tolist(),
                    ),
                    _format_macro_bool("source_proto_valid", losses.get("source_proto_valid", [])),
                    _format_macro_bool("target_proto_valid", losses.get("target_proto_valid", [])),
                )

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
