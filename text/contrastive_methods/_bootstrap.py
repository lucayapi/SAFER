"""Ajoute le code legacy contrastif au sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

from safer_core.paths import LEGACY_DIR, TEXT_ROOT

_LEGACY_CONTRASTIVE = LEGACY_DIR / "contrastive_method_v0"


def legacy_subdir(name: str) -> Path:
    mapping = {
        "batch_triplet": "batchTripplet",
        "softtriple": "Softriple",
        "supcon": "Supcon",
    }
    folder = mapping.get(name, name)
    path = _LEGACY_CONTRASTIVE / folder
    if not path.is_dir():
        raise FileNotFoundError(f"Dossier legacy manquant : {path}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def warn_deprecated_prompt(use_prompt: bool) -> None:
    from safer_core.text_columns import warn_if_prompt_enabled

    warn_if_prompt_enabled(use_prompt)
