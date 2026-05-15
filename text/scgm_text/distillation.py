"""Self-distillation utilities (aligned with official SCGM-G DistillKL)."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from scgm_text.scgm_embedding_model import SCGMEmbeddingNet


class DistillKL(nn.Module):
    """KL divergence for distillation (official SCGM-G)."""

    def __init__(self, temperature: float) -> None:
        super().__init__()
        self.T = float(temperature)

    def forward(self, y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        return F.kl_div(p_s, p_t, reduction="batchmean") * (self.T**2)


class EMATeacher:
    """Exponential moving average teacher for self-distillation."""

    def __init__(self, model: "SCGMEmbeddingNet", decay: float = 0.999) -> None:
        self.decay = decay
        self.teacher = copy.deepcopy(model)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: "SCGMEmbeddingNet") -> None:
        for t_param, s_param in zip(self.teacher.parameters(), student.parameters()):
            t_param.data.mul_(self.decay).add_(s_param.data, alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.teacher.state_dict()

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.teacher.load_state_dict(state)


def build_teacher(
    student: "SCGMEmbeddingNet",
    teacher_mode: str,
    ema_decay: float = 0.999,
) -> Optional[EMATeacher]:
    mode = str(teacher_mode).strip().lower()
    if mode == "none":
        return None
    if mode == "ema":
        return EMATeacher(student, decay=ema_decay)
    if mode == "previous_epoch":
        teacher = EMATeacher(student, decay=0.0)
        return teacher
    raise ValueError(f"Unknown teacher_mode: {teacher_mode!r} (expected ema, previous_epoch, none)")


@torch.no_grad()
def teacher_logits(
    teacher_model: "SCGMEmbeddingNet",
    features: torch.Tensor,
    batch_y: torch.Tensor,
    tau: float,
    norm_type: str = "logit",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_model.eval()
    return teacher_model.forward_to_logits(features, batch_y, tau=tau, norm_type=norm_type)


def snapshot_teacher_from_student(ema: EMATeacher, student: "SCGMEmbeddingNet") -> None:
    """Hard copy (previous_epoch mode) at epoch boundary."""
    ema.teacher.load_state_dict(student.state_dict())
    ema.teacher.eval()
