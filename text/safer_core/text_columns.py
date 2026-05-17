"""Colonnes texte standard."""

from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd

DEFAULT_TEXT_COL = "sentence"
FALLBACK_TEXT_COLS = ("sentence", "accident_summary", "text")


def resolve_text_col(df: pd.DataFrame, text_col: Optional[str] = None) -> str:
    if text_col and text_col in df.columns:
        return text_col
    for candidate in FALLBACK_TEXT_COLS:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Colonne texte introuvable. Fournir text_col ou ajouter une de {FALLBACK_TEXT_COLS}."
    )


def warn_if_prompt_enabled(use_prompt: bool) -> None:
    if use_prompt:
        warnings.warn(
            "Prompt-based training is deprecated in the main SAFER pipeline. "
            "Use text_col='sentence' without instruction prefixes.",
            UserWarning,
            stacklevel=2,
        )
