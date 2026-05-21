"""Contexte sectoriel des corpus pour les prompts OpenAI (macro_transfer)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from safer_core.io import load_yaml
from safer_core.paths import CONFIG_DIR, TEXT_ROOT, resolve_repo_path
from safer_core.test_corpus import load_test_corpora_registry, resolve_test_corpus

DEFAULT_CORPUS_CONTEXT_YAML = CONFIG_DIR / "corpus_prompt_context.yaml"


@dataclass(frozen=True)
class CorpusPromptContext:
    """Métadonnées de contexte sectoriel pour un corpus."""

    context_id: str
    title: str
    description: str


def _normalize_context(context_id: str, raw: dict) -> CorpusPromptContext:
    if not isinstance(raw, dict):
        raise ValueError(f"Entrée invalide pour contexte corpus {context_id!r}")
    title = str(raw.get("title", "")).strip()
    description = str(raw.get("description", "")).strip()
    if not title and not description:
        raise ValueError(f"Contexte {context_id} : title ou description requis")
    return CorpusPromptContext(
        context_id=context_id,
        title=title or context_id,
        description=description,
    )


@lru_cache(maxsize=4)
def _load_context_registry(path_str: str) -> Dict[str, CorpusPromptContext]:
    data = load_yaml(Path(path_str))
    contexts = data.get("contexts") or {}
    if not isinstance(contexts, dict):
        raise ValueError("corpus_prompt_context.yaml : clé 'contexts' manquante ou invalide")
    out: Dict[str, CorpusPromptContext] = {}
    for cid, entry in contexts.items():
        key = str(cid).strip()
        out[key] = _normalize_context(key, entry)
    return out


def load_corpus_prompt_contexts(
    yaml_path: Optional[Path | str] = None,
    *,
    anchor: Optional[Path] = None,
) -> Dict[str, CorpusPromptContext]:
    """Charge tous les contextes depuis ``configs/corpus_prompt_context.yaml``."""
    root = anchor or TEXT_ROOT
    path = resolve_repo_path(yaml_path or DEFAULT_CORPUS_CONTEXT_YAML, repo_root=root)
    if not path.is_file():
        raise FileNotFoundError(f"Contextes corpus introuvables : {path}")
    return dict(_load_context_registry(str(path.resolve())))


def get_corpus_prompt_context(
    context_id: str,
    *,
    yaml_path: Optional[Path | str] = None,
    anchor: Optional[Path] = None,
) -> Optional[CorpusPromptContext]:
    """Retourne un contexte par identifiant ou ``None``."""
    cid = str(context_id).strip()
    if not cid:
        return None
    registry = load_corpus_prompt_contexts(yaml_path, anchor=anchor)
    return registry.get(cid)


def resolve_prompt_context_key(
    corpus_id: str,
    *,
    registry_path: Optional[Path | str] = None,
    anchor: Optional[Path] = None,
) -> Optional[str]:
    """
    Résout la clé ``prompt_context`` pour un corpus test.

    Défaut : l'identifiant du corpus lui-même.
    """
    root = anchor or TEXT_ROOT
    try:
        spec = resolve_test_corpus(
            corpus_id, registry_path=registry_path, anchor=root
        )
    except (ValueError, FileNotFoundError, KeyError):
        cid = str(corpus_id).strip()
        return cid or None
    reg = load_test_corpora_registry(registry_path, anchor=root)
    entry = (reg.get("corpora") or {}).get(spec.id) or {}
    if isinstance(entry, dict):
        key = entry.get("prompt_context")
        if key is not None and str(key).strip():
            return str(key).strip()
    return spec.id


def format_corpus_context_for_prompt(
    corpus_id: str,
    *,
    context_yaml_path: Optional[Path | str] = None,
    anchor: Optional[Path] = None,
    registry_path: Optional[Path | str] = None,
) -> Optional[str]:
    """
    Bloc texte corpus à injecter dans le prompt OpenAI.

    Retourne ``None`` si le contexte est introuvable.
    """
    key = resolve_prompt_context_key(
        corpus_id, registry_path=registry_path, anchor=anchor
    )
    if not key:
        return None
    spec = get_corpus_prompt_context(key, yaml_path=context_yaml_path, anchor=anchor)
    if spec is None:
        return None
    lines = [f"Corpus : {spec.title}"]
    if spec.description:
        lines.append(spec.description.strip())
    return "\n".join(lines)


__all__ = [
    "DEFAULT_CORPUS_CONTEXT_YAML",
    "CorpusPromptContext",
    "format_corpus_context_for_prompt",
    "get_corpus_prompt_context",
    "load_corpus_prompt_contexts",
    "resolve_prompt_context_key",
]
