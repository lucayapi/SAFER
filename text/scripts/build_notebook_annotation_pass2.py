#!/usr/bin/env python3
"""Génère annotation/annotate_pass2_ambiguous.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "annotation" / "annotate_pass2_ambiguous.ipynb"

import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from safer_core.notebook_bootstrap import NOTEBOOK_PATH_SETUP


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


def py(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


cells = [
    md(
        """
# Passe 2 — désambiguïsation avec le récit complet (v13)

Ce notebook complète la **passe 1 batch** (`v13_two_pass_ambiguity_context`, `pass_mode=pass1`) :

1. Charge les annotations passe 1 (`PASS1_RUN_ID`)
2. Affiche les **statistiques d'ambiguïté** (volume passe 2 estimé)
3. Filtre les unités ambiguës (`context_needed`, `ambiguous`, `alternative_label`)
4. Ré-annote en synchrone avec le **récit complet** (`accident_summary`)
5. Fusionne dans `*__annotated_final.xlsx`

**Prérequis** : `OPENAI_API_KEY` dans `text/.env` et une run batch passe 1 terminée (`ingest`).
"""
    ),
    py(NOTEBOOK_PATH_SETUP),
    py(
        """
# --- Paramètres passe 2 ---
from pathlib import Path

PASS1_RUN_ID = "btp_v13_pass1__gpt-5.4-mini__v13_two_pass_ambiguity_context__pass1__20260711T150000Z"
OUTPUT_BASENAME = "btp_v13_pass2"
OPENAI_MODEL = "gpt-5.4-mini"
REASONING_EFFORT = "medium"
PROMPT_VERSION = "v13_two_pass_ambiguity_context"
PASS_MODE = "pass2"
PROMPT_CACHE_KEY = None
USE_PROMPT_CACHE_KEY = True

SUMMARY_COL = "accident_summary"
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 4000
SAVE_EVERY = 25
MAX_RETRIES = 3
RETRY_BASE_SLEEP_SEC = 1.5
RATE_LIMIT_SLEEP_SEC = 30.0
MIN_DELAY_BETWEEN_CALLS_SEC = 0.5
SKIP_CACHE = False
DRY_RUN = False
RUN_ID = None

ANNOTATION_ROOT = TEXT_ROOT / "annotation"
print("PASS1_RUN_ID =", PASS1_RUN_ID)
print("Modèle :", OPENAI_MODEL, "| pass_mode :", PASS_MODE)
"""
    ),
    py(
        """
# --- Étape 1 : charger la passe 1 ---
import pandas as pd
from annotation.config import AnnotationConfig
from annotation.two_pass import (
    filter_pass2_candidates,
    load_pass1_annotated,
    pass2_ambiguity_overview,
    pass2_selection_stats,
)

pass1_df = load_pass1_annotated(
    PASS1_RUN_ID,
    annotation_root=ANNOTATION_ROOT,
    openai_model=OPENAI_MODEL,
    prompt_version=PROMPT_VERSION,
    pass_mode="pass1",
)
print(f"Passe 1 chargée : {len(pass1_df)} unités")
pass1_df.head(3)
"""
    ),
    py(
        """
# --- Étape 2 : statistiques ambiguïté (avant ré-annotation) ---
from IPython.display import display

overview = pass2_ambiguity_overview(pass1_df)
summary = overview["summary"]

print("=== Synthèse passe 2 ===")
print(f"Unités passe 1 annotées     : {summary['n_pass1_units']:,}")
print(f"  dont pred_ok              : {summary['n_pred_ok']:,}")
print(f"  dont pred_not_ok          : {summary['n_pred_not_ok']:,}")
print()
print(f"Marqueurs d'ambiguïté (passe 1) :")
print(f"  ambiguous=true            : {summary['n_ambiguous']:,}")
print(f"  context_needed=true       : {summary['n_context_needed']:,}")
print(f"  alternative_label != NONE : {summary['n_alternative_label']:,}")
print()
print(f"→ Candidats passe 2         : {summary['n_pass2_candidates']:,} "
      f"({summary['pct_pass2_candidates']:.2f} % des unités)")
print(f"→ Accidents concernés       : {summary['n_accidents_with_candidate']:,} "
      f"/ {summary['n_accidents']:,}")

if summary["n_pass2_candidates"] == 0:
    print("\\nAucune ré-annotation passe 2 nécessaire.")
else:
    est_calls = summary["n_pass2_candidates"]
    print(f"\\nAppels API estimés (passe 2) : ~{est_calls:,}")

if not overview["by_label"].empty:
    print("\\nCandidats par label passe 1 :")
    display(overview["by_label"])
if not overview["by_ambiguity_type"].empty:
    print("Candidats par type d'ambiguïté :")
    display(overview["by_ambiguity_type"])
if not overview["by_alternative_label"].empty:
    print("Candidats par alternative_label :")
    display(overview["by_alternative_label"])
if not overview["candidates_preview"].empty:
    print("Aperçu (10 premières unités candidates) :")
    display(overview["candidates_preview"])
"""
    ),
    py(
        """
# --- Étape 3 : filtrer les candidats passe 2 ---
df_pass2 = filter_pass2_candidates(pass1_df)
print(f"{len(df_pass2)} unités à ré-annoter sur {len(pass1_df)}")
df_pass2.head(5)
"""
    ),
    py(
        """
# --- Étape 4 : configuration passe 2 ---
cfg = AnnotationConfig(
    input_csv="btp_sentence_accidents.csv",
    output_basename=OUTPUT_BASENAME,
    openai_model=OPENAI_MODEL,
    prompt_version=PROMPT_VERSION,
    pass_mode=PASS_MODE,
    pass1_run_id=PASS1_RUN_ID,
    prompt_cache_key=PROMPT_CACHE_KEY,
    use_prompt_cache_key=USE_PROMPT_CACHE_KEY,
    n_accidents="all",
    units_per_accident="all",
    temperature=TEMPERATURE,
    reasoning_effort=REASONING_EFFORT,
    max_output_tokens=MAX_OUTPUT_TOKENS,
    save_every=SAVE_EVERY,
    max_retries=MAX_RETRIES,
    retry_base_sleep_sec=RETRY_BASE_SLEEP_SEC,
    rate_limit_sleep_sec=RATE_LIMIT_SLEEP_SEC,
    min_delay_between_calls_sec=MIN_DELAY_BETWEEN_CALLS_SEC,
    skip_cache=SKIP_CACHE,
    dry_run=DRY_RUN,
    summary_col=SUMMARY_COL,
    annotation_root=ANNOTATION_ROOT,
    run_id=RUN_ID,
)
print("run_id =", cfg.run_id)
print("outputs_dir =", cfg.outputs_dir)
"""
    ),
    py(
        """
# --- Étape 5 : annotation synchrone passe 2 ---
from annotation.runner import classify_dataframe_with_cache

final_df, meta = classify_dataframe_with_cache(
    df_pass2,
    cfg,
    pass1_df=pass1_df,
)
meta
"""
    ),
    py(
        """
# --- Étape 6 : statistiques après fusion ---
from annotation.aggregate import summarize_predictions

summary_df = summarize_predictions(final_df)
display(summary_df)

if "pred_reannotated" in final_df.columns:
    n_reannotated = int(final_df["pred_reannotated"].fillna(False).sum())
    print(f"Unités réannotées (fusion finale) : {n_reannotated}")

label_changes = 0
if "pred_label" in pass1_df.columns and "pred_reannotated" in final_df.columns:
    merged_keys = final_df[final_df["pred_reannotated"].fillna(False)]
    print(f"Lignes pass2 traitées avec succès : {len(merged_keys)}")
"""
    ),
]

NB_PATH.write_text(
    json.dumps(
        {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python"},
            },
            "cells": cells,
        },
        ensure_ascii=False,
        indent=1,
    ),
    encoding="utf-8",
)
print(f"Écrit : {NB_PATH}")
