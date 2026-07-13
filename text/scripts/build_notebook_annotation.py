#!/usr/bin/env python3
"""Génère annotation/annotate_factual_units.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "annotation" / "annotate_factual_units.ipynb"

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
# Annotation d'unités factuelles (API OpenAI)

Notebook pas-à-pas pour annoter des récits d'accidents :
- macro-label **A0 / A1 / B / C**
- variables **injury_mentioned**, **hospitalized**, **fatal**
- cache **JSONL** local pour reprise sans tout refaire
- **prompt caching OpenAI** (`prompt_cache_key` + `cached_tokens` dans les logs)
- snapshots XLSX périodiques

**Prérequis** : `OPENAI_API_KEY` dans `text/.env` (voir `.env.example`).
Placez votre CSV dans `annotation/data/`.
"""
    ),
    py(NOTEBOOK_PATH_SETUP),
    py(
        """
# --- Paramètres ---
from pathlib import Path

INPUT_CSV = "mon_corpus.csv"          # relatif à annotation/data/
OUTPUT_BASENAME = "btp_v10_gpt5_mini"
OPENAI_MODEL = "gpt-5.4-mini"
REASONING_EFFORT = "medium"
PROMPT_VERSION = "v10_macro_labels_independent_outcomes"
PROMPT_CACHE_KEY = None               # None → safer-annotation:{PROMPT_VERSION}
USE_PROMPT_CACHE_KEY = True           # prompt caching OpenAI (préfixe system identique)

N_ACCIDENTS = 10                      # int ou "all"
UNITS_PER_ACCIDENT = "all"            # int ou "all"
ACCIDENT_SAMPLE_SEED = 42

TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 4000          # budget raisonnement (medium) + JSON
SAVE_EVERY = 50
MAX_RETRIES = 3
RETRY_BASE_SLEEP_SEC = 1.5
RATE_LIMIT_SLEEP_SEC = 30.0
MIN_DELAY_BETWEEN_CALLS_SEC = 0.5

SKIP_CACHE = False                    # True = ignorer le cache JSONL
DRY_RUN = False                       # True = prévisualiser sans appeler l'API
SUMMARY_COL = "accident_summary"      # ou summary_accident

RUN_ID = None                          # ex. reprendre une run existante

ANNOTATION_ROOT = TEXT_ROOT / "annotation"
print("ANNOTATION_ROOT =", ANNOTATION_ROOT)
print("Modèle :", OPENAI_MODEL, "| reasoning :", REASONING_EFFORT)
"""
    ),
    py(
        """
# --- Étape 1 : chargement CSV ---
import pandas as pd
from annotation.config import AnnotationConfig

cfg = AnnotationConfig(
    input_csv=INPUT_CSV,
    output_basename=OUTPUT_BASENAME,
    openai_model=OPENAI_MODEL,
    prompt_version=PROMPT_VERSION,
    prompt_cache_key=PROMPT_CACHE_KEY,
    use_prompt_cache_key=USE_PROMPT_CACHE_KEY,
    n_accidents=N_ACCIDENTS,
    units_per_accident=UNITS_PER_ACCIDENT,
    accident_sample_seed=ACCIDENT_SAMPLE_SEED,
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

input_path = cfg.resolved_input_path
print(f"Chargement : {input_path}")
df_raw = pd.read_csv(input_path)
print(f"Lignes totales : {len(df_raw):,}")
if "fact_id" not in df_raw.columns:
    print("⚠ fact_id absent — les clés de cache utiliseront ROWIDX_*")
cfg.validate_input_columns(list(df_raw.columns))
df_raw.head(3)
"""
    ),
    py(
        """
# --- Étape 2 : sous-échantillonnage accidents / unités ---
from annotation.runner import prepare_annotation_frame
from annotation.sampling import sampling_stats

df_work = prepare_annotation_frame(cfg, df_raw)
stats = sampling_stats(df_work)
print("Sous-ensemble :")
for k, v in stats.items():
    print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
if DRY_RUN:
    print("DRY_RUN=True → aucun appel API ne sera effectué.")
else:
    print(f"Appels API estimés : {stats['estimated_api_calls']:,}")
df_work.head(5)
"""
    ),
    py(
        """
# --- Étape 3 : chemins de sortie ---
print("run_id =", cfg.run_id)
print("outputs_dir =", cfg.outputs_dir)
cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
"""
    ),
    py(
        """
# --- Étape 4 : inspection du cache ---
from annotation.cache import get_output_paths, load_cache, make_cache_key

jsonl_path, snapshot_path, annotated_path, summary_path, accident_path = get_output_paths(
    cfg.outputs_dir,
    model_id=cfg.openai_model,
    prompt_version=cfg.prompt_version,
)
cache = {} if cfg.skip_cache else load_cache(jsonl_path)
n_hits = sum(
    1
    for idx, row in df_work.iterrows()
    if make_cache_key(row, row_idx=idx) in cache
)
print(f"Cache JSONL : {jsonl_path}")
print(f"Entrées en cache : {len(cache):,}")
print(f"Hits attendus sur ce sous-ensemble : {n_hits:,} / {len(df_work):,}")
print(f"Nouveaux appels estimés : {len(df_work) - n_hits:,}")
"""
    ),
    py(
        """
# --- Étape 5 : annotation (tqdm) ---
from annotation.runner import classify_dataframe_with_cache

df_pred, meta = classify_dataframe_with_cache(df_work, cfg, show_errors=True)
print("Meta :", meta)
df_pred.head(3)
"""
    ),
    py(
        """
# --- Étape 6 : résumé des prédictions ---
from annotation.aggregate import summarize_predictions

summary_df = summarize_predictions(df_pred)
print(summary_df.to_string(index=False))
print(f"pred_ok : {df_pred['pred_ok'].sum():,} / {len(df_pred):,}")
if meta.get("total_tokens"):
    print(f"Tokens totaux (run) : {meta['total_tokens']:,}")
if meta.get("total_prompt_tokens"):
    print(f"Tokens prompt : {meta['total_prompt_tokens']:,}")
if meta.get("total_cached_tokens"):
    print(f"Tokens prompt cachés (OpenAI) : {meta['total_cached_tokens']:,}")
    if meta.get("prompt_cache_hit_rate") is not None:
        print(f"Taux cached/prompt : {meta['prompt_cache_hit_rate']:.1%}")
"""
    ),
    py(
        """
# --- Étape 7 : agrégation au niveau accident ---
from annotation.aggregate import aggregate_outcomes_by_accident
from annotation.export_io import save_annotation_table

accident_df = aggregate_outcomes_by_accident(df_pred)
save_annotation_table(accident_df, meta["accident_xlsx_path"])
print("Sauvegardé :", meta["accident_xlsx_path"])
accident_df.head(10)
"""
    ),
]

nb = {
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
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Écrit :", NB_PATH)
