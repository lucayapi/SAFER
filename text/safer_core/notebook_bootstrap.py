"""Snippet réutilisable pour la première cellule code des notebooks (sys.path)."""

from __future__ import annotations

# À coller en tête de la première cellule — sans import safer_core avant ce bloc.
NOTEBOOK_PATH_SETUP = """import os
import sys
from pathlib import Path


def _notebook_find_text_root(start: Path) -> Path:
    here = start.resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "safer_core" / "paths.py").is_file():
            return candidate
        nested = candidate / "text"
        if (nested / "safer_core" / "paths.py").is_file():
            return nested
    raise FileNotFoundError(
        "Racine text/ introuvable (safer_core/paths.py). "
        "Ouvrez Jupyter depuis le dossier text/ ou SAFER/."
    )


TEXT_ROOT = _notebook_find_text_root(Path.cwd())
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))
os.chdir(TEXT_ROOT)
"""
