"""Résolution méthode / checkpoint / sortie pour Frozen Source Prototypes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from safer_core.paths import resolve_repo_path
from safer_core.test_corpus import macro_transfer_output_dir

FSP_ENCODER_METHODS: tuple[str, ...] = (
    "scgm_text",
    "softtriple",
    "supcon",
    "batch_triplet",
    "raw_embedding",
)

FspEncoderName = Literal[
    "scgm_text",
    "softtriple",
    "supcon",
    "batch_triplet",
    "raw_embedding",
]

# Alias rétrocompat (lecture seule) pour anciens runs.
FSP_METHOD_ALIASES: dict[str, str] = {
    "scgm": "scgm_text",
    "raw": "raw_embedding",
}

# Dossiers legacy sous frozen_source_prototypes/ (avant renommage scgm_text / raw_embedding).
FSP_LEGACY_OUTPUT_DIR_ALIASES: dict[str, str] = {
    "scgm_text": "scgm",
    "raw_embedding": "raw",
}


def normalize_fsp_base_method(method: str) -> str:
    m = str(method).strip()
    if m.startswith("frozen_source_prototypes/"):
        m = m.split("/", 1)[1]
    return FSP_METHOD_ALIASES.get(m, m)


def validate_fsp_base_method(method: str) -> FspEncoderName:
    m = normalize_fsp_base_method(method)
    if m not in FSP_ENCODER_METHODS:
        raise ValueError(
            f"Encodeur FSP non supporté : {method!r}. Attendu : {', '.join(FSP_ENCODER_METHODS)}"
        )
    return m  # type: ignore[return-value]


def fsp_output_method_key(base_method: str) -> str:
    base = validate_fsp_base_method(base_method)
    return f"frozen_source_prototypes/{base}"


def _fsp_has_transfer_artifacts(root: Path) -> bool:
    transfer = root / "transfer"
    if not transfer.is_dir():
        return False
    return (transfer / "metrics.json").is_file() or (
        transfer / "target_macro_predictions.csv"
    ).is_file()


def resolve_fsp_output_dir(
    corpus: str,
    base_method: str,
    *,
    anchor: Path,
    output_dir: Optional[str] = None,
) -> Path:
    if output_dir:
        return resolve_repo_path(str(output_dir), repo_root=anchor)

    base = validate_fsp_base_method(base_method)
    canonical = macro_transfer_output_dir(
        fsp_output_method_key(base),
        corpus,
        anchor=anchor,
    )
    if _fsp_has_transfer_artifacts(canonical):
        return canonical

    legacy_suffix = FSP_LEGACY_OUTPUT_DIR_ALIASES.get(base)
    if legacy_suffix:
        legacy = macro_transfer_output_dir(
            f"frozen_source_prototypes/{legacy_suffix}",
            corpus,
            anchor=anchor,
        )
        if _fsp_has_transfer_artifacts(legacy):
            return legacy

    return canonical


def resolve_fsp_checkpoint(
    base_method: str,
    model_cfg: dict[str, Any],
    checkpoints_block: Optional[dict[str, Any]] = None,
    *,
    explicit_checkpoint: Optional[str] = None,
    base_method_overridden: bool = False,
) -> Optional[str]:
    """
    Priorité : explicit_checkpoint > (si override CLI) checkpoints[base_method]
    > model.checkpoint_path > checkpoints[base_method].
    ``raw_embedding`` → None.
    """
    base = validate_fsp_base_method(base_method)
    if base == "raw_embedding":
        return None

    if explicit_checkpoint:
        return str(explicit_checkpoint)

    block = checkpoints_block or {}
    if base_method_overridden:
        ckpt = block.get(base)
        if ckpt:
            return str(ckpt)
        raise ValueError(
            f"Checkpoint manquant pour {base!r} "
            f"(base_method surchargé : utiliser checkpoints.{base})"
        )

    for key in ("checkpoint_path", "checkpoint"):
        ckpt = model_cfg.get(key)
        if ckpt:
            return str(ckpt)

    ckpt = block.get(base)
    if ckpt:
        return str(ckpt)

    raise ValueError(
        f"Checkpoint manquant pour {base!r} "
        f"(model.checkpoint_path ou checkpoints.{base})"
    )


def resolve_fsp_method_display_name(
    base_method: str,
    *,
    cfg_display: Optional[str] = None,
    model_display: Optional[str] = None,
) -> str:
    if cfg_display:
        return str(cfg_display)
    if model_display:
        return str(model_display)
    base = validate_fsp_base_method(base_method)
    if base == "scgm_text":
        return "SCGM + prototypes source gelés"
    if base == "raw_embedding":
        return "Embedding brut + prototypes source"
    return f"{base} + prototypes source gelés"
