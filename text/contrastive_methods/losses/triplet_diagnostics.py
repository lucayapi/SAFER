"""
Diagnostics Batch Hard Triplet (monitoring d'entraînement).

Interprétation rapide
---------------------
- **loss ≈ ln(2) ≈ 0,693** et **triplet_gap ≈ 0** longtemps : les hard positives et hard
  negatives sont à distance comparable — la loss soft-margin est peu informative.
- **loss plate mais triplet_gap qui augmente** : la géométrie progresse malgré une loss
  quasi constante (régime typique de BatchHardSoftMarginTripletLoss).
- **embedding_norm_mean** qui explose ou s'effondre : instabilité numérique / absence de
  normalisation effective.
- **pairwise_distance_mean → 0** : risque de collapse (tous les embeddings proches).
- **active_triplet_ratio → 1** longtemps (soft-margin) : presque tous les triplets restent
  « actifs » (hard neg pas clairement plus loin que hard pos).
- **active_triplet_ratio → 0** (loss hard avec marge) : la plupart des triplets sont déjà
  satisfaits.
- **batch_label_counts** avec peu d'exemples par label : composition de batch défavorable
  pour le batch hard mining (vérifier GROUP_BY_LABEL et batch_size).

Référence : « In Defense of the Triplet Loss for Person Re-Identification ».
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainerCallback

TRIPLET_DIAG_CSV_COLUMNS: List[str] = [
    "global_step",
    "epoch",
    "loss",
    "mean_hard_pos_dist",
    "mean_hard_neg_dist",
    "triplet_gap",
    "embedding_norm_mean",
    "embedding_norm_std",
    "pairwise_distance_mean",
    "pairwise_distance_std",
    "active_triplet_ratio",
    "batch_size_effective",
    "n_valid_anchors",
    "batch_label_counts",
    "distance_metric",
    "loss_type",
    "margin",
    "learning_rate",
]


def pairwise_distance_matrix(
    embeddings: torch.Tensor,
    distance_metric: str,
) -> torch.Tensor:
    """Matrice D [B, B] (cosine : 1 - cos sur embeddings L2-norm ; euclidean : cdist)."""
    metric = (distance_metric or "euclidean").strip().lower()
    if metric == "cosine":
        z = F.normalize(embeddings, p=2, dim=1)
        return 1.0 - z @ z.T
    if metric in ("euclidean", "eucledian"):
        return torch.cdist(embeddings, embeddings, p=2)
    raise ValueError(
        f"distance_metric inconnue : {distance_metric!r} (attendu : cosine, euclidean)"
    )


def _validate_labels(labels: torch.Tensor) -> torch.Tensor:
    if labels.dim() != 1:
        raise ValueError(f"labels doit être un tenseur 1D, reçu shape {tuple(labels.shape)}")
    return labels.long().contiguous()


def _batch_label_counts_dict(labels: torch.Tensor) -> Dict[str, int]:
    labels_cpu = labels.detach().cpu().long()
    if labels_cpu.numel() == 0:
        return {}
    max_label = int(labels_cpu.max().item())
    counts = torch.bincount(labels_cpu, minlength=max_label + 1)
    return {str(i): int(c) for i, c in enumerate(counts.tolist()) if c > 0}


def _nan_stats() -> Dict[str, Any]:
    nan = float("nan")
    return {
        "mean_hard_pos_dist": nan,
        "mean_hard_neg_dist": nan,
        "triplet_gap": nan,
        "embedding_norm_mean": nan,
        "embedding_norm_std": nan,
        "pairwise_distance_mean": nan,
        "pairwise_distance_std": nan,
        "active_triplet_ratio": nan,
        "n_valid_anchors": 0,
    }


def compute_batch_hard_triplet_stats(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    distance_metric: str,
    *,
    margin: Optional[float] = None,
    soft_margin: bool = True,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """
    Statistiques batch-hard (sans gradient).

    Retourne des floats Python ; NaN si aucune ancre valide (positif et négatif dans le batch).
    """
    with torch.no_grad():
        labels = _validate_labels(labels)
        b = int(embeddings.size(0))
        d = pairwise_distance_matrix(embeddings, distance_metric)
        eye = torch.eye(b, dtype=torch.bool, device=embeddings.device)
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = same & ~eye
        neg_mask = ~same
        has_pos = pos_mask.any(dim=1)
        has_neg = neg_mask.any(dim=1)
        valid = has_pos & has_neg

        off_diag = d[~eye]
        pairwise_mean = float(off_diag.mean().item()) if off_diag.numel() else float("nan")
        pairwise_std = (
            float(off_diag.std(unbiased=False).item()) if off_diag.numel() > 1 else float("nan")
        )

        norms = embeddings.norm(p=2, dim=1)
        norm_mean = float(norms.mean().item())
        norm_std = float(norms.std(unbiased=False).item()) if b > 1 else 0.0

        out: Dict[str, Any] = {
            "batch_size_effective": b,
            "batch_label_counts": _batch_label_counts_dict(labels),
            "embedding_norm_mean": norm_mean,
            "embedding_norm_std": norm_std,
            "pairwise_distance_mean": pairwise_mean,
            "pairwise_distance_std": pairwise_std,
        }

        if not bool(valid.any()):
            out.update(_nan_stats())
            return out

        d_pos_all = d.masked_fill(~pos_mask, float("-inf"))
        hardest_pos, _ = d_pos_all.max(dim=1)
        d_neg_all = d.masked_fill(~neg_mask, float("inf"))
        hardest_neg, _ = d_neg_all.min(dim=1)

        hp = hardest_pos[valid]
        hn = hardest_neg[valid]
        mean_pos = float(hp.mean().item())
        mean_neg = float(hn.mean().item())
        gap = mean_neg - mean_pos

        if soft_margin:
            active = (hp >= hn - float(eps)).float().mean().item()
        else:
            m = float(margin if margin is not None else 0.0)
            active = (hp - hn + m > 0).float().mean().item()

        out.update(
            {
                "mean_hard_pos_dist": mean_pos,
                "mean_hard_neg_dist": mean_neg,
                "triplet_gap": gap,
                "active_triplet_ratio": float(active),
                "n_valid_anchors": int(valid.sum().item()),
            }
        )
        return out


def batch_hard_triplet_loss_value(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    distance_metric: str,
    *,
    margin: Optional[float] = None,
    soft_margin: bool = True,
) -> torch.Tensor:
    """Loss batch-hard (differentiable) sur les ancres valides uniquement."""
    labels = _validate_labels(labels)
    d = pairwise_distance_matrix(embeddings, distance_metric)
    b = labels.size(0)
    eye = torch.eye(b, dtype=torch.bool, device=embeddings.device)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = same & ~eye
    neg_mask = ~same
    valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)

    if not bool(valid.any()):
        return embeddings.sum() * 0.0

    d_pos_all = d.masked_fill(~pos_mask, float("-inf"))
    hardest_pos, _ = d_pos_all.max(dim=1)
    d_neg_all = d.masked_fill(~neg_mask, float("inf"))
    hardest_neg, _ = d_neg_all.min(dim=1)

    hp = hardest_pos[valid]
    hn = hardest_neg[valid]
    if soft_margin:
        return torch.log1p(torch.exp(hp - hn)).mean()
    m = float(margin if margin is not None else 0.0)
    return F.relu(hp - hn + m).mean()


def append_triplet_diagnostics_csv(row: Dict[str, Any], output_path: Path) -> None:
    """Ajoute une ligne au CSV de diagnostics (crée l'en-tête si besoin)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(row)
    if isinstance(serializable.get("batch_label_counts"), dict):
        serializable["batch_label_counts"] = json.dumps(
            serializable["batch_label_counts"], sort_keys=True
        )
    write_header = not output_path.is_file() or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRIPLET_DIAG_CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: serializable.get(k) for k in TRIPLET_DIAG_CSV_COLUMNS})


def format_triplet_diag_console(row: Dict[str, Any]) -> str:
    """Ligne console compacte pour un pas de diagnostic."""
    labels = row.get("batch_label_counts")
    if isinstance(labels, dict):
        labels_s = json.dumps(labels, sort_keys=True)
    else:
        labels_s = str(labels)

    def _f(key: str, ndigits: int = 3) -> str:
        v = row.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "nan"
        try:
            return f"{float(v):.{ndigits}f}"
        except (TypeError, ValueError):
            return str(v)

    return (
        f"[TripletDiag step={row.get('global_step')} epoch={row.get('epoch')}] "
        f"loss={_f('loss', 4)} | "
        f"hard_pos={_f('mean_hard_pos_dist')} | hard_neg={_f('mean_hard_neg_dist')} | "
        f"gap={_f('triplet_gap')} | active={_f('active_triplet_ratio')} | "
        f"norm={_f('embedding_norm_mean')}±{_f('embedding_norm_std')} | "
        f"pairwise={_f('pairwise_distance_mean')}±{_f('pairwise_distance_std')} | "
        f"labels={labels_s}"
    )


class TripletDiagnosticsCallback(TrainerCallback):
    """Écrit CSV + log console tous les ``every_steps`` pas d'entraînement."""

    def __init__(
        self,
        loss_module: Any,
        output_path: Path,
        *,
        every_steps: int = 50,
        distance_metric: str,
        loss_type: str,
        margin: Optional[float] = None,
    ) -> None:
        self.loss_module = loss_module
        self.output_path = Path(output_path)
        self.every_steps = max(1, int(every_steps))
        self.distance_metric = distance_metric
        self.loss_type = loss_type
        self.margin = margin

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step <= 0 or step % self.every_steps != 0:
            return control
        pending = getattr(self.loss_module, "consume_pending_diagnostics", None)
        if pending is None:
            return control
        base = pending()
        if not base:
            return control

        loss_val = None
        for entry in reversed(state.log_history or []):
            if entry.get("loss") is not None and "eval" not in entry:
                try:
                    loss_val = float(entry["loss"])
                    break
                except (TypeError, ValueError):
                    pass

        lr = None
        for entry in reversed(state.log_history or []):
            if entry.get("learning_rate") is not None:
                try:
                    lr = float(entry["learning_rate"])
                    break
                except (TypeError, ValueError):
                    pass

        row = {
            **base,
            "global_step": step,
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "loss": loss_val,
            "distance_metric": self.distance_metric,
            "loss_type": self.loss_type,
            "margin": self.margin,
            "learning_rate": lr,
        }
        append_triplet_diagnostics_csv(row, self.output_path)
        print(format_triplet_diag_console(row), flush=True)
        return control
