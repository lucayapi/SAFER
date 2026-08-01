"""Configuration du pipeline d'annotation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional, Union

from annotation.prompts import build_artifact_slug
from annotation.usage import default_prompt_cache_key

IntOrAll = Union[int, Literal["all"]]
ReasoningEffort = Literal["minimal", "low", "medium", "high"]
PassMode = Literal["pass1", "pass2"]
VALID_REASONING_EFFORTS = ("minimal", "low", "medium", "high")
VALID_PASS_MODES = ("pass1", "pass2")


def _sanitize_slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name).strip()) or "run"


def _normalize_int_or_all(value: IntOrAll | str) -> IntOrAll:
    if isinstance(value, str) and value.strip().lower() == "all":
        return "all"
    return int(value)


def _normalize_reasoning_effort(value: str) -> ReasoningEffort:
    effort = str(value).strip().lower()
    if effort not in VALID_REASONING_EFFORTS:
        raise ValueError(
            f"reasoning_effort invalide : {value!r}. "
            f"Valeurs acceptées : {', '.join(VALID_REASONING_EFFORTS)}"
        )
    return effort  # type: ignore[return-value]


def _normalize_pass_mode(value: str) -> PassMode:
    mode = str(value).strip().lower()
    if mode not in VALID_PASS_MODES:
        raise ValueError(
            f"pass_mode invalide : {value!r}. "
            f"Valeurs acceptées : {', '.join(VALID_PASS_MODES)}"
        )
    return mode  # type: ignore[return-value]


@dataclass
class AnnotationConfig:
    input_csv: Path
    output_basename: str = "annotation_run"
    openai_model: str = "gpt-5.4-mini"
    prompt_version: str = "v10_macro_labels_independent_outcomes"
    prompt_cache_key: Optional[str] = None
    use_prompt_cache_key: bool = True
    n_accidents: IntOrAll = "all"
    units_per_accident: IntOrAll = "all"
    accident_sample_seed: int = 42
    accident_sample_frac: Optional[float] = None
    temperature: float = 0.0
    reasoning_effort: ReasoningEffort = "medium"
    max_output_tokens: int = 4000
    save_every: int = 50
    max_retries: int = 3
    retry_base_sleep_sec: float = 1.5
    rate_limit_sleep_sec: float = 30.0
    min_delay_between_calls_sec: float = 0.5
    skip_cache: bool = False
    dry_run: bool = False
    summary_col: str = "accident_summary"
    pass_mode: PassMode = "pass1"
    pass1_run_id: Optional[str] = None
    annotation_root: Optional[Path] = None
    run_id: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.n_accidents = _normalize_int_or_all(self.n_accidents)
        self.units_per_accident = _normalize_int_or_all(self.units_per_accident)
        self.reasoning_effort = _normalize_reasoning_effort(self.reasoning_effort)
        self.pass_mode = _normalize_pass_mode(self.pass_mode)
        if self.accident_sample_frac is not None:
            frac = float(self.accident_sample_frac)
            if not (0.0 < frac <= 1.0):
                raise ValueError(
                    f"accident_sample_frac doit être dans (0, 1], reçu : {frac!r}"
                )
            self.accident_sample_frac = frac
        self.input_csv = Path(self.input_csv)
        if self.annotation_root is None:
            self.annotation_root = Path(__file__).resolve().parent
        else:
            self.annotation_root = Path(self.annotation_root)
        if self.run_id is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            model_slug = _sanitize_slug(self.openai_model)
            prompt_slug = _sanitize_slug(self.prompt_version)
            pass_slug = _sanitize_slug(self.pass_mode)
            base = _sanitize_slug(self.output_basename)
            self.run_id = f"{base}__{model_slug}__{prompt_slug}__{pass_slug}__{ts}"
        if self.prompt_cache_key is None and self.use_prompt_cache_key:
            self.prompt_cache_key = default_prompt_cache_key(
                f"{self.prompt_version}:{self.pass_mode}"
            )
        if self.pass_mode == "pass2" and not self.pass1_run_id:
            raise ValueError(
                "pass_mode=pass2 requiert pass1_run_id (run batch passe 1 à fusionner)."
            )

    @property
    def artifact_slug(self) -> str:
        return build_artifact_slug(self.prompt_version, self.pass_mode)

    @property
    def effective_prompt_cache_key(self) -> Optional[str]:
        if not self.use_prompt_cache_key:
            return None
        return self.prompt_cache_key

    @property
    def data_dir(self) -> Path:
        return self.annotation_root / "data"

    @property
    def outputs_dir(self) -> Path:
        return self.annotation_root / "outputs" / self.run_id

    @property
    def resolved_input_path(self) -> Path:
        path = Path(self.input_csv)
        if path.is_absolute():
            return path
        candidate = self.data_dir / path
        if candidate.is_file():
            return candidate
        return path

    def validate_input_columns(self, columns: list[str]) -> None:
        missing = {"accident_id", "sentence"} - set(columns)
        if missing:
            raise ValueError(f"Colonnes obligatoires manquantes : {sorted(missing)}")
        if self.summary_col not in columns and "summary_accident" not in columns:
            raise ValueError(
                f"Colonne résumé introuvable : {self.summary_col!r} "
                "(attendu accident_summary ou summary_accident)."
            )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["input_csv"] = str(self.input_csv)
        data["annotation_root"] = str(self.annotation_root)
        data["resolved_input_path"] = str(self.resolved_input_path)
        data["outputs_dir"] = str(self.outputs_dir)
        data["effective_prompt_cache_key"] = self.effective_prompt_cache_key
        return data
