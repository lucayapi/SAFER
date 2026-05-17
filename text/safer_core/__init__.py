"""Utilitaires centraux du pipeline SAFER texte."""

from safer_core.paths import (
    CONFIG_DIR,
    DATASET_DIR,
    JOBS_DIR,
    METHOD_RESULTS_DIRS,
    NOTEBOOKS_DIR,
    PROJECT_ROOT,
    RESULTS_ROOT,
    TEXT_ROOT,
    ensure_comparisons_dirs,
    ensure_method_dirs,
    get_method_dir,
    resolve_output_dir,
)

__all__ = [
    "PROJECT_ROOT",
    "TEXT_ROOT",
    "DATASET_DIR",
    "CONFIG_DIR",
    "RESULTS_ROOT",
    "METHOD_RESULTS_DIRS",
    "JOBS_DIR",
    "NOTEBOOKS_DIR",
    "get_method_dir",
    "ensure_method_dirs",
    "ensure_comparisons_dirs",
    "resolve_output_dir",
]
