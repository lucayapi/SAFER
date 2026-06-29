"""Chemins de sortie supervised_macro_ft (train / transfert)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from safer_core.paths import resolve_repo_path
from safer_core.test_corpus import macro_transfer_output_dir, output_test_root

logger = logging.getLogger(__name__)


def supervised_macro_output_dir(method_slug: str, corpus: str, *, anchor: Path) -> Path:
    return macro_transfer_output_dir(method_slug, corpus, anchor=anchor)


def _has_supervised_macro_transfer_artifacts(root: Path) -> bool:
    transfer = root / "transfer"
    if not transfer.is_dir():
        return False
    return (transfer / "target_macro_predictions.csv").is_file() or (
        transfer / "metrics.json"
    ).is_file()


def resolve_supervised_macro_output_dir(
    method_slug: str,
    corpus: str,
    *,
    anchor: Path,
    output_dir: Optional[str] = None,
) -> Path:
    """Chemin canonique macro_transfer/<method>/, repli legacy output_test/<corpus>/<method>/."""
    if output_dir:
        return resolve_repo_path(str(output_dir), repo_root=anchor)

    canonical = supervised_macro_output_dir(method_slug, corpus, anchor=anchor)
    if _has_supervised_macro_transfer_artifacts(canonical):
        return canonical

    legacy = (output_test_root(anchor=anchor) / corpus / method_slug).resolve()
    if _has_supervised_macro_transfer_artifacts(legacy):
        logger.info("Artefacts transfert trouvés (legacy) : %s", legacy)
        return legacy

    return canonical


def supervised_macro_ft_output_dir(corpus: str, *, anchor: Path) -> Path:
    return resolve_supervised_macro_output_dir("supervised_macro_ft", corpus, anchor=anchor)
