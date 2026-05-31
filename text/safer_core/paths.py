"""Chemins centralisés pour le pipeline SAFER texte."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, Optional

METHOD_NAMES = (
    "raw_embedding",
    "scgm_text",
    "batch_triplet",
    "softtriple",
    "supcon",
)

METHOD_SUBDIRS = (
    "configs",
    "checkpoints",
    "embeddings",
    "assignments",
    "metrics",
    "figures",
    "topics",
    "logs",
)

COMPARISON_SUBDIRS = ("tables", "figures", "reports", "notebooks_exports")

_LEGACY_RUNS_PREFIX = "runs/"
_LEGACY_OUTPUTS_PREFIX = "outputs/"
_LEGACY_TO_OUTPUT: Dict[str, str] = {
    "runs/scgm_text_qwen06": "output/scgm_text",
    "runs/scgm_text_qwen06_notebook": "output/scgm_text",
    "outputs/bn_btp_from_scgm": "output/scgm_text/bn_results",
    "outputs/topic_comparison": "output/comparisons/topics_legacy",
    "resultats/scgm_text": "output/scgm_text",
    "resultats/raw_embedding": "output/raw_embedding",
    "resultats/comparisons": "output/comparisons",
}
_LEGACY_PATH_PREFIXES = (
    ("resultats_test/", "output_test/"),
    ("resultats/", "output/"),
)


def find_text_root(start: Optional[Path] = None) -> Path:
    """Racine ``text/`` (contient ``safer_core/`` ou ``scripts/``)."""
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "safer_core" / "paths.py").is_file():
            return candidate
        if (candidate / "text" / "safer_core" / "paths.py").is_file():
            return candidate / "text"
    return here


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Alias : racine du dépôt pipeline texte."""
    return find_text_root(start)


def resolve_repo_path(path: str | Path, repo_root: Path | None = None) -> Path:
    """Chemin absolu : si relatif, depuis ``repo_root`` ou racine du dépôt détectée."""
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    root = repo_root if repo_root is not None else find_repo_root()
    return (root / p).resolve()


TEXT_ROOT = find_text_root()
PROJECT_ROOT = TEXT_ROOT.parent if TEXT_ROOT.name == "text" else TEXT_ROOT
DATASET_DIR = TEXT_ROOT / "dataset"
CONFIG_DIR = TEXT_ROOT / "configs"
METHODS_CONFIG_DIR = CONFIG_DIR / "methods"
OUTPUT_ROOT = TEXT_ROOT / "output"
RESULTS_ROOT = OUTPUT_ROOT  # alias rétrocompat
JOBS_DIR = TEXT_ROOT / "jobs"
NOTEBOOKS_DIR = TEXT_ROOT / "notebooks"
LEGACY_DIR = TEXT_ROOT / "legacy"

METHOD_RESULTS_DIRS: Dict[str, Path] = {
    name: RESULTS_ROOT / name for name in METHOD_NAMES
}


def get_method_dir(method_name: str) -> Path:
    key = str(method_name).strip().lower().replace("-", "_")
    if key not in METHOD_RESULTS_DIRS:
        raise ValueError(f"Méthode inconnue : {method_name!r} (attendu : {METHOD_NAMES})")
    return METHOD_RESULTS_DIRS[key]


def ensure_method_dirs(
    method_name: str,
    *,
    extra: Optional[Iterable[str]] = None,
) -> Path:
    root = get_method_dir(method_name)
    names = set(METHOD_SUBDIRS)
    if extra:
        names.update(extra)
    if method_name in ("raw_embedding", "batch_triplet", "softtriple", "supcon"):
        names.discard("assignments")
    for sub in sorted(names):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def ensure_comparisons_dirs() -> Path:
    root = RESULTS_ROOT / "comparisons"
    for sub in COMPARISON_SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def _normalize_legacy_path(path: str | Path) -> Optional[Path]:
    p = str(path).replace("\\", "/").strip().rstrip("/")
    for legacy, target in _LEGACY_TO_OUTPUT.items():
        if p == legacy or p.startswith(legacy + "/"):
            suffix = p[len(legacy) :].lstrip("/")
            new = target + ("/" + suffix if suffix else "")
            return (TEXT_ROOT / new).resolve()
    for old_prefix, new_prefix in _LEGACY_PATH_PREFIXES:
        if p == old_prefix.rstrip("/") or p.startswith(old_prefix):
            suffix = p[len(old_prefix.rstrip("/")) :].lstrip("/")
            mapped = new_prefix.rstrip("/") + ("/" + suffix if suffix else "")
            return (TEXT_ROOT / mapped).resolve()
    if p.startswith(_LEGACY_RUNS_PREFIX) or p.startswith(_LEGACY_OUTPUTS_PREFIX):
        return None
    return None


def resolve_output_dir(
    method_name: str,
    output_dir: Optional[str | Path] = None,
    *,
    legacy_ok: bool = True,
) -> Path:
    """
    Résout le dossier de sortie d'une méthode.

    - défaut : ``text/output/<method>``
    - si ``runs/`` ou ``outputs/`` et ``SAFER_LEGACY_PATHS=1`` : warning + mapping connu
    """
    default = get_method_dir(method_name)
    if output_dir is None or str(output_dir).strip() == "":
        ensure_method_dirs(method_name)
        return default

    raw = str(output_dir).strip()
    p = Path(raw)
    if not p.is_absolute():
        p = (TEXT_ROOT / p).resolve()

    norm = str(p).replace("\\", "/")
    if legacy_ok and (
        norm.endswith("/" + _LEGACY_RUNS_PREFIX.rstrip("/"))
        or _LEGACY_RUNS_PREFIX in norm
        or _LEGACY_OUTPUTS_PREFIX in norm
        or os.environ.get("SAFER_LEGACY_PATHS", "").lower() in ("1", "true", "yes")
    ):
        mapped = _normalize_legacy_path(p)
        if mapped is not None:
            warnings.warn(
                f"Chemin legacy {output_dir!r} redirigé vers {mapped}. "
                "Préférez text/output/<method>/.",
                UserWarning,
                stacklevel=2,
            )
            ensure_method_dirs(method_name)
            return mapped
        if _LEGACY_RUNS_PREFIX in norm or _LEGACY_OUTPUTS_PREFIX in norm:
            warnings.warn(
                f"Chemin legacy non mappé : {output_dir!r}. "
                f"Utilisation de {default}.",
                UserWarning,
                stacklevel=2,
            )

    if p == default or str(p).startswith(str(RESULTS_ROOT)):
        ensure_method_dirs(method_name)
    return p


def method_checkpoints_dir(method_name: str) -> Path:
    return get_method_dir(method_name) / "checkpoints"


def method_metrics_dir(method_name: str) -> Path:
    return get_method_dir(method_name) / "metrics"


def method_logs_dir(method_name: str) -> Path:
    return get_method_dir(method_name) / "logs"


def layout_method_output(method_name: str, output_dir: Optional[str | Path] = None) -> Dict[str, Path]:
    """Résout la racine méthode et crée configs/checkpoints/embeddings/..."""
    root = resolve_output_dir(method_name, output_dir)
    ensure_method_dirs(method_name)
    return {
        "root": root,
        "configs": root / "configs",
        "checkpoints": root / "checkpoints",
        "embeddings": root / "embeddings",
        "assignments": root / "assignments",
        "metrics": root / "metrics",
        "figures": root / "figures",
        "topics": root / "topics",
        "logs": root / "logs",
    }
