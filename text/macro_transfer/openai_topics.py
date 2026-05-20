"""Enrichissement OpenAI des tables topics intra-macro."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, List, Optional, Union

import pandas as pd
from tqdm.auto import tqdm

from scgm_text.openai_theme_labels import (
    _fallback_row_labels,
    _get_client,
    _one_row,
    load_openai_dotenv,
)


def enrich_topics_openai(
    themes_csv: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    n_example_texts: int = 5,
    summary_words_min: int = 3,
    summary_words_max: int = 12,
    client: Any = None,
    show_progress: bool = True,
    skip_on_error: bool = True,
    request_timeout: Optional[float] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Enrichit ``themes_by_macro.csv`` (colonnes macro, topic_id, top_sentences, …).
    """
    load_openai_dotenv()
    themes_path = Path(themes_csv)
    if not themes_path.is_file():
        raise FileNotFoundError(str(themes_path))
    frame = pd.read_csv(themes_path)
    required = {"macro", "topic_id", "n_units", "top_words", "top_sentences"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {themes_path}: {sorted(missing)}")

    out_path = Path(output_csv) if output_csv else themes_path.with_name(
        themes_path.stem + "_openai.csv"
    )
    cli = client or _get_client()
    n_ex = max(1, min(int(n_example_texts), 20))
    lo = max(1, int(summary_words_min))
    hi = max(lo, int(summary_words_max))

    titles: List[str] = []
    summaries: List[str] = []
    kw_strings: List[str] = []
    rows = list(frame.iterrows())
    if max_rows is not None:
        rows = rows[: max(0, int(max_rows))]
    iterator = tqdm(rows, desc="OpenAI topics intra-macro", unit="topic") if show_progress else rows
    failures = 0
    for _, row in iterator:
        pseudo = {
            "z_id": row.get("topic_id"),
            "dominant_macro": row.get("macro"),
            "n_units": row.get("n_units"),
            "top_words": row.get("top_words"),
            "top_sentences": row.get("top_sentences"),
        }
        try:
            parsed = _one_row(
                cli,
                model=model,
                temperature=temperature,
                row=pseudo,
                n_example_texts=n_ex,
                summary_words_min=lo,
                summary_words_max=hi,
                request_timeout=request_timeout,
            )
        except Exception as exc:
            if not skip_on_error:
                raise
            failures += 1
            parsed = _fallback_row_labels(pseudo, summary_words_min=lo, summary_words_max=hi)
            warnings.warn(
                f"macro={row.get('macro')} topic_id={row.get('topic_id')}: {type(exc).__name__}",
                stacklevel=2,
            )
        titles.append(parsed["theme_title"])
        summaries.append(parsed["theme_summary"])
        kw_strings.append(parsed["theme_keywords"])

    enriched = frame.copy()
    enriched["theme_title"] = titles
    enriched["theme_summary"] = summaries
    enriched["theme_keywords"] = kw_strings
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False)
    return enriched
