"""Self-distillation (legacy). Not imported by the official end2end SCGM pipeline."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from scgm_text.scgm_text_model import SCGMTextModel


class DistillKL(nn.Module):
    def __init__(self, temperature: float) -> None:
        super().__init__()
        self.T = float(temperature)

    def forward(self, y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        return F.kl_div(p_s, p_t, reduction="batchmean") * (self.T**2)


class EMATeacher:
    def __init__(self, model: "SCGMTextModel", decay: float = 0.999) -> None:
        self.decay = decay
        self.teacher = copy.deepcopy(model)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: "SCGMTextModel") -> None:
        for t_param, s_param in zip(self.teacher.parameters(), student.parameters()):
            t_param.data.mul_(self.decay).add_(s_param.data, alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.teacher.state_dict()

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.teacher.load_state_dict(state)
