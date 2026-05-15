"""Résolution des chemins relatifs à la racine du dépôt (indépendant du cwd, ex. notebooks/)."""

from __future__ import annotations

from pathlib import Path


def find_repo_root() -> Path:
    """Racine du pipeline texte (dossier contenant ``topic_eval/``)."""
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "topic_eval" / "__init__.py").is_file():
            return candidate
        if (candidate / "text" / "topic_eval" / "__init__.py").is_file():
            return candidate / "text"
    return here


def resolve_repo_path(path: str | Path, repo_root: Path | None = None) -> Path:
    """Chemin absolu : si relatif, depuis ``repo_root`` ou racine du dépôt détectée."""
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    root = repo_root if repo_root is not None else find_repo_root()
    return (root / p).resolve()
