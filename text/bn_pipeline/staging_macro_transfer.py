"""Staging des exports macro_transfer pour le pipeline BN (corpus test)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from safer_core.paths import find_repo_root, resolve_repo_path
from safer_core.test_corpus import bn_staging_dir, macro_transfer_output_dir

TOPICS_SUBDIR = "topics_bertopic"

_COPY_EMBEDDINGS = (
    ("embeddings/prob_z_x.npy", "pt_z_target.npy"),
    ("embeddings/prob_y_z.npy", "pt_y_given_z.npy"),
)


def macro_transfer_root(
    method: str = "scgm_text",
    corpus_id: Optional[str] = None,
    *,
    repo_root: Optional[Path] = None,
) -> Path:
    root = repo_root or find_repo_root()
    return macro_transfer_output_dir(method, corpus_id, anchor=root)


def topics_subdir() -> str:
    """Sous-dossier topics intra-macro (BERTopic)."""
    return TOPICS_SUBDIR


def stage_bn_exports_from_macro_transfer(
    method: str = "scgm_text",
    corpus_id: Optional[str] = None,
    *,
    output_dir: Optional[str | Path] = None,
    repo_root: Optional[Path] = None,
) -> Path:
    """
    Copie les artefacts macro_transfer vers ``bn_staging/staging/bn_exports/``.

    Entrées attendues sous ``output_test/<corpus>/macro_transfer/<method>/`` :
    ``transfer/metadata_with_macro_probs.csv``, ``topics_bertopic/assignments.csv``,
    ``topics_bertopic/themes_by_macro.csv`` (colonnes ``theme_label``, ``top_words``),
    ``embeddings/prob_*.npy``.
    """
    root = repo_root or find_repo_root()
    mt = macro_transfer_root(method, corpus_id, repo_root=root)
    if output_dir is not None:
        out_root = resolve_repo_path(output_dir, root)
    else:
        out_root = bn_staging_dir(corpus_id, anchor=root)
    exports = out_root / "staging" / "bn_exports"
    exports.mkdir(parents=True, exist_ok=True)

    transfer_meta = mt / "transfer" / "metadata_with_macro_probs.csv"
    if not transfer_meta.is_file():
        raise FileNotFoundError(f"metadata_with_macro_probs.csv manquant : {transfer_meta}")
    shutil.copy2(transfer_meta, exports / "metadata_with_predictions.csv")

    topics_dir = mt / topics_subdir()
    assign_src = topics_dir / "assignments.csv"
    themes_src = topics_dir / "themes_by_macro.csv"
    if not assign_src.is_file():
        raise FileNotFoundError(f"assignments.csv manquant : {assign_src}")
    shutil.copy2(assign_src, exports / "macro_topic_assignments.csv")
    if themes_src.is_file():
        shutil.copy2(themes_src, exports / "themes_by_macro.csv")

    emb_dir = mt / "embeddings"
    for rel_src, dst_name in _COPY_EMBEDDINGS:
        src = emb_dir / Path(rel_src).name
        if src.is_file():
            shutil.copy2(src, exports / dst_name)

    manifest = {
        "method": method,
        "corpus_id": corpus_id,
        "macro_transfer_dir": str(mt),
        "topic_source": "bertopic",
        "bn_exports": str(exports),
    }
    (exports / "staging_manifest.json").write_text(
        __import__("json").dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return exports
