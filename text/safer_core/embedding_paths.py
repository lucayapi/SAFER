"""Canonical, backbone-isolated locations for frozen embedding exports."""

from __future__ import annotations

import re
from pathlib import Path

from safer_core.paths import TEXT_ROOT


_KNOWN_BACKBONES = {
    "qwen/qwen3-embedding-0.6b": "qwen3",
    "intfloat/multilingual-e5-large": "multilingual_e5_large",
}


def backbone_storage_id(backbone_name: str) -> str:
    """Return a stable safe directory name, without relying on a CSV registry."""
    name = str(backbone_name).strip().lower()
    if name in _KNOWN_BACKBONES:
        return _KNOWN_BACKBONES[name]
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def embedding_export_path(
    corpus_id: str,
    *,
    backbone_name: str,
    output_root: str | Path = "embeddings",
    backbone_id: str | None = None,
    anchor: Path = TEXT_ROOT,
) -> Path:
    """Return ``<output_root>/<backbone_id>/<corpus>.csv`` under ``text/``."""
    root = Path(output_root)
    if not root.is_absolute():
        root = anchor / root
    identifier = str(backbone_id or backbone_storage_id(backbone_name)).strip()
    if not identifier or "/" in identifier or "\\" in identifier:
        raise ValueError(f"Invalid backbone storage identifier: {identifier!r}")
    return root / identifier / f"{str(corpus_id)}.csv"
