"""Configuration du pipeline d'annotation Batch OpenAI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from annotation.config import AnnotationConfig
from scgm_text.utils_io import load_yaml_config


@dataclass
class BatchAnnotationConfig(AnnotationConfig):
    completion_window: str = "24h"
    batch_endpoint: str = "/v1/chat/completions"
    status_poll_interval_sec: float = 60.0
    max_requests_per_batch: int = 12_000
    max_batch_retries: int = 1

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["completion_window"] = self.completion_window
        data["batch_endpoint"] = self.batch_endpoint
        data["status_poll_interval_sec"] = self.status_poll_interval_sec
        data["max_requests_per_batch"] = self.max_requests_per_batch
        data["max_batch_retries"] = self.max_batch_retries
        return data


def _coerce_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none"}:
        return None
    return text


def _batch_config_from_mapping(
    raw: dict[str, Any],
    *,
    annotation_root: Optional[Path] = None,
) -> BatchAnnotationConfig:
    root = annotation_root or Path(__file__).resolve().parent
    return BatchAnnotationConfig(
        input_csv=Path(raw.get("input_csv", "mon_corpus.csv")),
        output_basename=str(raw.get("output_basename", "annotation_batch")),
        openai_model=str(raw.get("openai_model", "gpt-5.4-mini")),
        prompt_version=str(raw.get("prompt_version", "v10_macro_labels_independent_outcomes")),
        prompt_cache_key=_coerce_optional_str(raw.get("prompt_cache_key")),
        use_prompt_cache_key=bool(raw.get("use_prompt_cache_key", True)),
        n_accidents=raw.get("n_accidents", "all"),  # type: ignore[arg-type]
        units_per_accident=raw.get("units_per_accident", "all"),  # type: ignore[arg-type]
        accident_sample_seed=int(raw.get("accident_sample_seed", 42)),
        temperature=float(raw.get("temperature", 0.0)),
        reasoning_effort=str(raw.get("reasoning_effort", "medium")),
        max_output_tokens=int(raw.get("max_output_tokens", 4000)),
        summary_col=str(raw.get("summary_col", "accident_summary")),
        pass_mode=str(raw.get("pass_mode", "pass1")),
        pass1_run_id=_coerce_optional_str(raw.get("pass1_run_id")),
        skip_cache=bool(raw.get("skip_cache", False)),
        annotation_root=root,
        run_id=_coerce_optional_str(raw.get("run_id")),
        completion_window=str(raw.get("completion_window", "24h")),
        batch_endpoint=str(raw.get("batch_endpoint", "/v1/chat/completions")),
        status_poll_interval_sec=float(raw.get("status_poll_interval_sec", 60.0)),
        max_requests_per_batch=int(raw.get("max_requests_per_batch", 12_000)),
        max_batch_retries=int(raw.get("max_batch_retries", 1)),
        extra=dict(raw.get("extra") or {}),
    )


def load_batch_config(
    config_path: str | Path,
    *,
    overrides: Optional[dict[str, Any]] = None,
    annotation_root: Optional[Path] = None,
) -> BatchAnnotationConfig:
    """Charge un YAML et construit une ``BatchAnnotationConfig``."""
    raw = load_yaml_config(str(config_path))
    if overrides:
        raw = {**raw, **{k: v for k, v in overrides.items() if v is not None}}
    return _batch_config_from_mapping(raw, annotation_root=annotation_root)


def load_batch_config_from_run(
    run_id: str,
    *,
    annotation_root: Optional[Path] = None,
    overrides: Optional[dict[str, Any]] = None,
) -> BatchAnnotationConfig:
    """Recharge la config exacte utilisée lors du submit (``run_config.json``)."""
    root = annotation_root or Path(__file__).resolve().parent
    outputs_dir = resolve_run_outputs_dir(root, run_id)
    path = outputs_dir / "run_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"run_config.json introuvable pour la run : {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if overrides:
        raw = {**raw, **{k: v for k, v in overrides.items() if v is not None}}
    return _batch_config_from_mapping(raw, annotation_root=root)


def align_cfg_for_existing_run(cfg: BatchAnnotationConfig) -> BatchAnnotationConfig:
    """Aligne download/ingest sur la config sauvegardée de la run, si disponible."""
    run_config_path = cfg.outputs_dir / "run_config.json"
    if not run_config_path.is_file():
        return cfg
    run_id = cfg.run_id or cfg.outputs_dir.name
    runtime_overrides: dict[str, Any] = {}
    if cfg.skip_cache:
        runtime_overrides["skip_cache"] = True
    return load_batch_config_from_run(
        run_id,
        annotation_root=cfg.annotation_root,
        overrides=runtime_overrides or None,
    )


def resolve_run_outputs_dir(annotation_root: Path, run_id: str) -> Path:
    return annotation_root / "outputs" / run_id
