"""Registre des prompts d'annotation."""

from __future__ import annotations

from typing import Any, Callable, Mapping

V13_PROMPT_VERSION = "v13_two_pass_ambiguity_context"
V10_PROMPT_VERSION = "v10_macro_labels_independent_outcomes"

KNOWN_PROMPT_VERSIONS = (V10_PROMPT_VERSION, V13_PROMPT_VERSION)


def is_v13_prompt(prompt_version: str) -> bool:
    version = str(prompt_version).strip()
    return version == V13_PROMPT_VERSION or version.startswith("v13_")


def build_artifact_slug(prompt_version: str, pass_mode: str = "pass1") -> str:
    """Slug fichiers cache/batch : évite collision pass1 vs pass2."""
    return f"{prompt_version}__{pass_mode}"


def resolve_prompt_bundle(
    prompt_version: str,
    *,
    pass_mode: str = "pass1",
    summary_col: str = "accident_summary",
) -> dict[str, Any]:
    """Charge le bundle prompt selon version et mode de passe."""
    if is_v13_prompt(prompt_version):
        from annotation.prompts.v13_two_pass_ambiguity_context import get_prompt_bundle

        return get_prompt_bundle(summary_col=summary_col, pass_mode=pass_mode)

    if prompt_version in {V10_PROMPT_VERSION} or prompt_version.startswith(("v10_", "v11_", "v12_")):
        from annotation.prompts.v10_macro_labels_independent_outcomes import (
            LABELS,
            MENTION_VALUES,
            PROMPT_VERSION,
            SYSTEM_PROMPT,
            build_user_prompt,
        )

        return {
            "prompt_version": PROMPT_VERSION,
            "pass_mode": "pass1",
            "system_prompt": SYSTEM_PROMPT,
            "build_user_prompt": lambda row, first_pass_annotation=None: build_user_prompt(
                row,
                summary_col=summary_col,
            ),
            "labels": LABELS,
            "mention_values": MENTION_VALUES,
            "should_run_second_pass": lambda _annotation: False,
        }

    raise ValueError(
        f"prompt_version inconnu : {prompt_version!r}. "
        f"Versions connues : {', '.join(KNOWN_PROMPT_VERSIONS)}"
    )


def empty_prediction_fields(prompt_version: str) -> dict[str, None]:
    """Champs pred_* à None pour les réponses en erreur."""
    base = {
        "pred_label": None,
        "pred_injury_mentioned": None,
        "pred_hospitalized": None,
        "pred_fatal": None,
        "pred_confidence": None,
        "pred_justification": None,
        "pred_context_used": None,
        "pred_ok": False,
    }
    if is_v13_prompt(prompt_version):
        base.update(
            {
                "pred_ambiguous": None,
                "pred_context_needed": None,
                "pred_alternative_label": None,
                "pred_ambiguity_type": None,
                "pred_ambiguity_reason": None,
            }
        )
    return base
