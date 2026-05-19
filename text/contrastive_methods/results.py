"""Résultats d'un run contrastif."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class TrainingResult:
    embeddings_path: Path
    output_root: Path
    val_geometry: Dict[str, Any] = field(default_factory=dict)
    best_eta2_macro_balanced_perc: float = float("nan")
    train_log_path: Optional[Path] = None
