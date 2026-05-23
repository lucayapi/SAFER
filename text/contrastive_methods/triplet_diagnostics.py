"""
Diagnostics Batch Hard Triplet (monitoring entraînement).

Interprétation rapide :
- loss ≈ log(2) et triplet_gap ≈ 0 longtemps : hard positives et hard negatives équidistants.
- loss plate mais triplet_gap qui augmente : progrès géométrique malgré une loss peu informative.
- embedding_norm_mean qui explose ou s'effondre : stabilité / normalisation.
- pairwise_distance_mean → 0 : risque de collapse des embeddings.
- active_triplet_ratio ≈ 1 longtemps (soft-margin) : presque tous les triplets restent difficiles.
- active_triplet_ratio → 0 (hard margin) : beaucoup de triplets déjà satisfaits.
- batch_label_counts avec peu d'exemples par label : batch mal composé pour le hard mining.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

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


def _normalize_distance_metric(name: str) -> str:
    key = (name or "euclidean").strip().lower()
    if key in ("euclidean", "eucledian"):
        return "euclidean"
    if key == "cosine":
        return "cosine"
    raise ValueError(f"distance_metric inconnue : {name!r} (attendu cosine | euclidean)")


def pairwise_distance_matrix(
    embeddings: torch.Tensor,
    distance_metric: str,
) -> torch.Tensor:
    """
    Matrice D de taille B×B.

    cosine : L2-normalize puis D_ij = 1 - cos(z_i, z_j)
    euclidean : D_ij = ||z_i - z_j||_2
    """
    metric = _normalize_distance_metric(distance_metric)
    z = embeddings
    if metric == "cosine":
        z = F.normalize(z, p=2, dim=-1)
        sim = z @ z.t()
        return 1.0 - sim
    return torch.cdist(z, z, p=2)


def _batch_hard_mining(
    pairwise_dist: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hard pos/neg par ancre (même masquage que Sentence Transformers BatchHardTripletLoss)."""
    from sentence_transformers.losses import BatchHardTripletLoss

    labels = labels.contiguous().view(-1)
    batch_size = labels.shape[0]
    device = pairwise_dist.device

    mask_pos = BatchHardTripletLoss.get_anchor_positive_triplet_mask(labels).float()
    mask_neg = BatchHardTripletLoss.get_anchor_negative_triplet_mask(labels).float()

    anchor_positive_dist, _ = (pairwise_dist * mask_pos).max(1, keepdim=True)
    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_neg)
    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

    hardest_positive_dist = anchor_positive_dist.squeeze(1)
    hardest_negative_dist = hardest_negative_dist.squeeze(1)

    has_pos = mask_pos.sum(1) > 0
    has_neg = mask_neg.sum(1) > 0
    valid = has_pos & has_neg

    return hardest_positive_dist, hardest_negative_dist, valid


def triplet_loss_from_hard_distances(
    d_pos: torch.Tensor,
    d_neg: torch.Tensor,
    *,
    soft_margin: bool = True,
    margin: Optional[float] = None,
) -> torch.Tensor:
    """Soft : log1p(exp(d+ - d-)) ; Hard : relu(d+ - d- + m)."""
    diff = d_pos - d_neg
    if soft_margin:
        return torch.log1p(torch.exp(diff)).mean()
    m = float(margin if margin is not None else 5.0)
    return F.relu(diff + m).mean()


@dataclass
class BatchHardTripletStats:
    mean_hard_pos_dist: float
    mean_hard_neg_dist: float
    triplet_gap: float
    embedding_norm_mean: float
    embedding_norm_std: float
    pairwise_distance_mean: float
    pairwise_distance_std: float
    active_triplet_ratio: float
    batch_size_effective: int
    n_valid_anchors: int
    batch_label_counts: Dict[str, int]
    distance_metric: str
    loss_type: str
    margin: Optional[float]
    d_pos: torch.Tensor = field(repr=False)
    d_neg: torch.Tensor = field(repr=False)
    valid_mask: torch.Tensor = field(repr=False)


def _label_counts_dict(labels: torch.Tensor) -> Dict[str, int]:
    labels_list = labels.detach().cpu().tolist()
    counts: Dict[str, int] = {}
    for lab in labels_list:
        key = str(int(lab))
        counts[key] = counts.get(key, 0) + 1
    return counts


def compute_batch_hard_triplet_stats(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    distance_metric: str,
    soft_margin: bool = True,
    margin: Optional[float] = None,
    eps: float = 1e-6,
) -> BatchHardTripletStats:
    """
    Statistiques de monitoring (sans modifier le graphe pour les scalaires exportés).

    Les tenseurs d_pos/d_neg/valid_mask sont conservés pour le calcul de loss différentiable.
    """
    metric = _normalize_distance_metric(distance_metric)
    labels = labels.contiguous().view(-1).long()
    if labels.dim() != 1:
        raise ValueError("labels doit être un tenseur 1D d'entiers")

    b = int(embeddings.shape[0])
    d_pos, d_neg, valid = _batch_hard_mining(
        pairwise_distance_matrix(embeddings, metric), labels
    )

    norms = embeddings.norm(p=2, dim=1)
    d_mat = pairwise_distance_matrix(embeddings, metric)
    off_diag = d_mat[~torch.eye(b, dtype=torch.bool, device=d_mat.device)]

    n_valid = int(valid.sum().item())
    loss_type = "soft_margin" if soft_margin else "hard"

    def _nan() -> float:
        return float("nan")

    if n_valid == 0:
        return BatchHardTripletStats(
            mean_hard_pos_dist=_nan(),
            mean_hard_neg_dist=_nan(),
            triplet_gap=_nan(),
            embedding_norm_mean=float(norms.mean().item()),
            embedding_norm_std=float(norms.std(unbiased=False).item()) if b > 1 else 0.0,
            pairwise_distance_mean=float(off_diag.mean().item()) if off_diag.numel() else _nan(),
            pairwise_distance_std=float(off_diag.std(unbiased=False).item()) if off_diag.numel() > 1 else 0.0,
            active_triplet_ratio=_nan(),
            batch_size_effective=b,
            n_valid_anchors=0,
            batch_label_counts=_label_counts_dict(labels),
            distance_metric=metric,
            loss_type=loss_type,
            margin=margin,
            d_pos=d_pos,
            d_neg=d_neg,
            valid_mask=valid,
        )

    d_pos_v = d_pos[valid]
    d_neg_v = d_neg[valid]
    mean_pos = float(d_pos_v.mean().item())
    mean_neg = float(d_neg_v.mean().item())
    gap = mean_neg - mean_pos

    diff = d_pos_v - d_neg_v
    if soft_margin:
        active = (diff >= -float(eps)).float().mean().item()
    else:
        m = float(margin if margin is not None else 5.0)
        active = (diff + m > 0).float().mean().item()

    return BatchHardTripletStats(
        mean_hard_pos_dist=mean_pos,
        mean_hard_neg_dist=mean_neg,
        triplet_gap=gap,
        embedding_norm_mean=float(norms.mean().item()),
        embedding_norm_std=float(norms.std(unbiased=False).item()) if b > 1 else 0.0,
        pairwise_distance_mean=float(off_diag.mean().item()),
        pairwise_distance_std=float(off_diag.std(unbiased=False).item()) if off_diag.numel() > 1 else 0.0,
        active_triplet_ratio=float(active),
        batch_size_effective=b,
        n_valid_anchors=n_valid,
        batch_label_counts=_label_counts_dict(labels),
        distance_metric=metric,
        loss_type=loss_type,
        margin=margin,
        d_pos=d_pos,
        d_neg=d_neg,
        valid_mask=valid,
    )


def _mono_label_batch_error_message(stats: BatchHardTripletStats) -> str:
    counts = json.dumps(stats.batch_label_counts, sort_keys=True)
    return (
        "[BATCH ERROR] BatchTriplet received a mono-label batch; no valid negative exists. "
        f"batch_label_counts={counts}, n_valid_anchors={stats.n_valid_anchors}."
    )


def raise_on_invalid_triplet_batch(stats: BatchHardTripletStats) -> None:
    if stats.n_valid_anchors > 0:
        return
    msg = _mono_label_batch_error_message(stats)
    print(msg, flush=True)
    raise RuntimeError(msg)


def stats_to_log_row(
    stats: BatchHardTripletStats,
    *,
    loss: float,
    global_step: int,
    epoch: Optional[float] = None,
    learning_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """Ligne plate pour CSV / console."""
    return {
        "global_step": global_step,
        "epoch": epoch,
        "loss": loss,
        "mean_hard_pos_dist": stats.mean_hard_pos_dist,
        "mean_hard_neg_dist": stats.mean_hard_neg_dist,
        "triplet_gap": stats.triplet_gap,
        "embedding_norm_mean": stats.embedding_norm_mean,
        "embedding_norm_std": stats.embedding_norm_std,
        "pairwise_distance_mean": stats.pairwise_distance_mean,
        "pairwise_distance_std": stats.pairwise_distance_std,
        "active_triplet_ratio": stats.active_triplet_ratio,
        "batch_size_effective": stats.batch_size_effective,
        "n_valid_anchors": stats.n_valid_anchors,
        "batch_label_counts": json.dumps(stats.batch_label_counts, sort_keys=True),
        "distance_metric": stats.distance_metric,
        "loss_type": stats.loss_type,
        "margin": stats.margin,
        "learning_rate": learning_rate,
    }


class TripletDiagnosticLogger:
    """Append CSV + log console tous les ``every_steps`` forwards."""

    def __init__(
        self,
        output_path: Union[str, Path],
        *,
        every_steps: int = 50,
    ) -> None:
        self.output_path = Path(output_path)
        self.every_steps = max(1, int(every_steps))
        self._forward_count = 0
        self._global_step = 0
        self._epoch: Optional[float] = None
        self._learning_rate: Optional[float] = None
        self._header_written = self.output_path.is_file() and self.output_path.stat().st_size > 0

    def set_training_context(
        self,
        *,
        global_step: Optional[int] = None,
        epoch: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ) -> None:
        if global_step is not None:
            self._global_step = int(global_step)
        if epoch is not None:
            self._epoch = float(epoch)
        if learning_rate is not None:
            self._learning_rate = float(learning_rate)

    def maybe_log(self, stats: BatchHardTripletStats, loss_value: float) -> None:
        self._forward_count += 1
        if self._forward_count % self.every_steps != 0:
            return

        if stats.n_valid_anchors == 0:
            print(_mono_label_batch_error_message(stats), flush=True)

        row = stats_to_log_row(
            stats,
            loss=float(loss_value),
            global_step=self._global_step,
            epoch=self._epoch,
            learning_rate=self._learning_rate,
        )
        self._append_csv(row)
        self._print_console(row)

    def _append_csv(self, row: Dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self._header_written
        with open(self.output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRIPLET_DIAG_CSV_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
                self._header_written = True
            out = {k: row.get(k) for k in TRIPLET_DIAG_CSV_COLUMNS}
            for k, v in out.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    out[k] = ""
            writer.writerow(out)
            f.flush()

    @staticmethod
    def _fmt_num(value: Any, *, precision: int = 4) -> str:
        if value is None:
            return "nan"
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "nan"
        if math.isnan(v) or math.isinf(v):
            return "nan"
        return f"{v:.{precision}f}"

    def _print_console(self, row: Dict[str, Any]) -> None:
        epoch_s = (
            self._fmt_num(row.get("epoch"), precision=2)
            if row.get("epoch") is not None
            else "?"
        )
        labels = row.get("batch_label_counts", "{}")
        print(
            f"[TripletDiag step={row['global_step']} epoch={epoch_s}] "
            f"loss={self._fmt_num(row.get('loss'))} | "
            f"hard_pos={self._fmt_num(row.get('mean_hard_pos_dist'))} | "
            f"hard_neg={self._fmt_num(row.get('mean_hard_neg_dist'))} | "
            f"gap={self._fmt_num(row.get('triplet_gap'))} | "
            f"active={self._fmt_num(row.get('active_triplet_ratio'))} | "
            f"norm={self._fmt_num(row.get('embedding_norm_mean'), precision=2)}"
            f"±{self._fmt_num(row.get('embedding_norm_std'), precision=2)} | "
            f"pairwise={self._fmt_num(row.get('pairwise_distance_mean'))}"
            f"±{self._fmt_num(row.get('pairwise_distance_std'))} | "
            f"labels={labels}",
            flush=True,
        )
