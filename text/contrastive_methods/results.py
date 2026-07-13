"""Résultats d'un run contrastif."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TrainingResult:
    embeddings_path: Path
    output_root: Path
    best_train_loss: float = float("nan")
    train_wall_time_sec: float = float("nan")
    train_log_path: Optional[Path] = None
