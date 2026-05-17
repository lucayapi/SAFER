"""Résolution des chemins relatifs à la racine du dépôt (indépendant du cwd, ex. notebooks/)."""

from __future__ import annotations

from pathlib import Path

from safer_core.paths import find_text_root as _find_text_root


def find_repo_root() -> Path:
    """Racine du pipeline texte (dossier contenant ``topic_eval/``)."""
    return _find_text_root()


def resolve_repo_path(path: str | Path, repo_root: Path | None = None) -> Path:
    """Chemin absolu : si relatif, depuis ``repo_root`` ou racine du dépôt détectée."""
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    root = repo_root if repo_root is not None else find_repo_root()
    return (root / p).resolve()
