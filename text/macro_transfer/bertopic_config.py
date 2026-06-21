"""Chargement / fusion de la config BERTopic intra-macro partagée."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from safer_core.io import load_yaml
from safer_core.paths import resolve_repo_path

DEFAULT_BERTOPIC_SHARED_REL = "configs/bertopic_macro_shared.yaml"


def default_bertopic_shared_path(*, anchor: Optional[Path] = None) -> Path:
    root = Path(anchor) if anchor is not None else Path(__file__).resolve().parents[1]
    return root / DEFAULT_BERTOPIC_SHARED_REL


def load_bertopic_macro_shared(*, anchor: Optional[Path] = None) -> Dict[str, Any]:
    """Charge ``configs/bertopic_macro_shared.yaml``."""
    path = default_bertopic_shared_path(anchor=anchor)
    if not path.is_file():
        raise FileNotFoundError(f"Config BERTopic partagée introuvable : {path}")
    return load_yaml(path)


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], val)
        else:
            out[key] = deepcopy(val)
    return out


def resolve_bertopic_run_config(
    cfg: Dict[str, Any],
    *,
    anchor: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Fusionne la config partagée avec les surcharges du run.

    Clé optionnelle dans le YAML du run : ``bertopic_shared`` (chemin relatif au repo).
    Retourne (bertopic, topics_export, topic_judge).
    """
    shared_rel = cfg.get("bertopic_shared") or DEFAULT_BERTOPIC_SHARED_REL
    shared_path = resolve_repo_path(str(shared_rel), repo_root=anchor)
    shared = load_yaml(shared_path) if shared_path.is_file() else {}

    bertopic = _deep_merge_dict(
        dict(shared.get("bertopic") or {}),
        dict(cfg.get("bertopic") or {}),
    )
    topics_export = _deep_merge_dict(
        dict(shared.get("topics_export") or {}),
        dict(cfg.get("topics_export") or {}),
    )
    topic_judge = _deep_merge_dict(
        dict(shared.get("topic_judge") or {}),
        dict(cfg.get("topic_judge") or {}),
    )
    return bertopic, topics_export, topic_judge


def enrich_run_config_bertopic(
    cfg: Dict[str, Any],
    *,
    anchor: Optional[Path] = None,
) -> Dict[str, Any]:
    """Injecte ``bertopic``, ``topics_export`` et ``topic_judge`` fusionnés."""
    out = dict(cfg)
    bertopic, topics_export, topic_judge = resolve_bertopic_run_config(cfg, anchor=anchor)
    out["bertopic"] = bertopic
    out["topics_export"] = topics_export
    out["topic_judge"] = topic_judge
    return out
