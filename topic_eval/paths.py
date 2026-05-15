"""Résolution des chemins relatifs à la racine du dépôt (indépendant du cwd, ex. notebooks/)."""

from __future__ import annotations

from pathlib import Path


def find_repo_root() -> Path:
    """Répertoire contenant le package ``topic_eval/`` (remonte depuis ``cwd``)."""
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        marker = candidate / "topic_eval" / "__init__.py"
        if marker.is_file():
            return candidate
    return here


def resolve_repo_path(path: str | Path, repo_root: Path | None = None) -> Path:
    """Chemin absolu : si relatif, depuis ``repo_root`` ou racine du dépôt détectée."""
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    root = repo_root if repo_root is not None else find_repo_root()
    return (root / p).resolve()
