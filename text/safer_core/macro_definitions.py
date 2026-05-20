"""Définitions sémantiques des macros A0–C (chargement YAML + formatage prompt OpenAI)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from safer_core.io import load_yaml
from safer_core.paths import CONFIG_DIR, TEXT_ROOT, resolve_repo_path

DEFAULT_MACROS_YAML = CONFIG_DIR / "accident_macros.yaml"
MACRO_ORDER = ("A0", "A1", "B", "C")


@dataclass(frozen=True)
class MacroDefinition:
    """Métadonnées d'une macro accidentologique."""

    macro_id: str
    title: str
    question: str
    description: str
    label_guidance: str


def _normalize_entry(macro_id: str, raw: dict) -> MacroDefinition:
    if not isinstance(raw, dict):
        raise ValueError(f"Entrée invalide pour macro {macro_id!r}")
    title = str(raw.get("title", "")).strip()
    question = str(raw.get("question", "")).strip()
    description = str(raw.get("description", "")).strip()
    label_guidance = str(raw.get("label_guidance", "")).strip()
    if not title or not question or not label_guidance:
        raise ValueError(f"Macro {macro_id} : title, question et label_guidance requis")
    return MacroDefinition(
        macro_id=macro_id,
        title=title,
        question=question,
        description=description,
        label_guidance=label_guidance,
    )


@lru_cache(maxsize=4)
def _load_registry(path_str: str) -> Dict[str, MacroDefinition]:
    data = load_yaml(Path(path_str))
    macros = data.get("macros") or {}
    if not isinstance(macros, dict):
        raise ValueError("accident_macros.yaml : clé 'macros' manquante ou invalide")
    out: Dict[str, MacroDefinition] = {}
    for macro_id, entry in macros.items():
        mid = str(macro_id).strip()
        out[mid] = _normalize_entry(mid, entry)
    return out


def load_macro_definitions(
    yaml_path: Optional[Path | str] = None,
    *,
    anchor: Optional[Path] = None,
) -> Dict[str, MacroDefinition]:
    """Charge toutes les définitions depuis ``configs/accident_macros.yaml``."""
    root = anchor or TEXT_ROOT
    path = resolve_repo_path(yaml_path or DEFAULT_MACROS_YAML, repo_root=root)
    if not path.is_file():
        raise FileNotFoundError(f"Définitions macros introuvables : {path}")
    return dict(_load_registry(str(path.resolve())))


def get_macro_definition(
    macro_id: str,
    *,
    yaml_path: Optional[Path | str] = None,
    anchor: Optional[Path] = None,
) -> Optional[MacroDefinition]:
    """Retourne la définition d'une macro ou ``None`` si l'identifiant est inconnu."""
    mid = str(macro_id).strip()
    if not mid:
        return None
    registry = load_macro_definitions(yaml_path, anchor=anchor)
    return registry.get(mid)


def format_macro_context_for_prompt(
    macro_id: str,
    *,
    include_description: bool = True,
    yaml_path: Optional[Path | str] = None,
    anchor: Optional[Path] = None,
) -> Optional[str]:
    """
    Bloc texte à injecter dans le prompt utilisateur OpenAI.

    Retourne ``None`` si la macro est absente ou inconnue.
    """
    spec = get_macro_definition(macro_id, yaml_path=yaml_path, anchor=anchor)
    if spec is None:
        return None
    lines = [
        f"Macro : {spec.macro_id} — {spec.title}",
        f"Question centrale : {spec.question}",
    ]
    if include_description and spec.description:
        lines.append(f"Description : {spec.description}")
    lines.append(f"Consigne pour le libellé : {spec.label_guidance}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_MACROS_YAML",
    "MACRO_ORDER",
    "MacroDefinition",
    "format_macro_context_for_prompt",
    "get_macro_definition",
    "load_macro_definitions",
]
