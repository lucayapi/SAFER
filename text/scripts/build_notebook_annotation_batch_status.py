#!/usr/bin/env python3
"""Génère annotation/check_batch_status.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "annotation" / "check_batch_status.ipynb"

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
# Suivi batch annotation OpenAI

Notebook léger pour vérifier si un job Batch est terminé et inspecter les fichiers locaux.

**Prérequis** : `OPENAI_API_KEY` dans `text/.env` pour rafraîchir le statut via l'API.

Workflow CLI complet :
```bash
python scripts/run_annotation_batch.py submit --config configs/annotation_batch.yaml
python scripts/run_annotation_batch.py status --run-id <RUN_ID>
python scripts/run_annotation_batch.py download --run-id <RUN_ID>
python scripts/run_annotation_batch.py ingest --run-id <RUN_ID>
```
"""
    ),
    py(NOTEBOOK_PATH_SETUP),
    py(
        """
# --- Paramètres ---
# Remplacez par le RUN_ID affiché après submit (dossier sous annotation/outputs/).
RUN_ID = "btp_v13_pass1__gpt-5.4-mini__v13_two_pass_ambiguity_context__pass1__20260711T155029Z"
REFRESH_FROM_API = True   # False = lecture locale batch_state.json uniquement

ANNOTATION_ROOT = TEXT_ROOT / "annotation"
OUTPUTS_DIR = ANNOTATION_ROOT / "outputs" / RUN_ID
BATCH_STATE_PATH = OUTPUTS_DIR / "batch_state.json"
BATCH_OUTPUT_PATH = OUTPUTS_DIR / "batch_output.jsonl"
BATCH_ERRORS_PATH = OUTPUTS_DIR / "batch_errors.jsonl"

print("RUN_ID =", RUN_ID)
print("OUTPUTS_DIR =", OUTPUTS_DIR)
if not OUTPUTS_DIR.is_dir():
    raise FileNotFoundError(
        f"Dossier run introuvable : {OUTPUTS_DIR}\\n"
        "Vérifiez RUN_ID (liste : annotation/outputs/)."
    )
"""
    ),
    py(
        """
# --- État local ---
import json
from pathlib import Path

state = {}
if BATCH_STATE_PATH.is_file():
    state = json.loads(BATCH_STATE_PATH.read_text(encoding="utf-8"))
    print("Statut (local) :", state.get("status"))
    print("Batch ID :", state.get("batch_id"))
    print("Soumis :", state.get("submitted_at"))
    print("Terminé :", state.get("completed_at"))
    counts = state.get("request_counts") or {}
    total = counts.get("total") or 0
    completed = counts.get("completed") or 0
    failed = counts.get("failed") or 0
    print(f"Progression : {completed}/{total} complétées, {failed} échecs")
    if total:
        pct = 100.0 * completed / total
        print(f"Avancement : {pct:.1f}%")
    state
else:
    print(f"batch_state.json introuvable : {BATCH_STATE_PATH}")
    if BATCH_OUTPUT_PATH.is_file():
        print("→ batch_output.jsonl est déjà présent : vous pouvez lancer l'ingest directement.")
    else:
        print("→ Exécutez submit/status via la CLI, ou corrigez RUN_ID.")
"""
    ),
    py(
        """
# --- Rafraîchir depuis l'API OpenAI ---
from annotation.batch_config import load_batch_config_from_run
from annotation.batch_client import list_batch_chunks, refresh_batch_state

cfg = load_batch_config_from_run(RUN_ID, annotation_root=ANNOTATION_ROOT)

if REFRESH_FROM_API:
    state = refresh_batch_state(cfg)
    chunks = list_batch_chunks(state)
    counts = state.get("request_counts") or {}
    print("Statut global (API) :", state.get("status"))
    print(
        f"Progression API : {counts.get('completed', 0)}/{counts.get('total', 0)} "
        f"(failed={counts.get('failed', 0)})"
    )
    for chunk in chunks:
        ccounts = chunk.get("request_counts") or {}
        print(
            f"  - {chunk.get('batch_id')} : {chunk.get('status')} "
            f"({ccounts.get('completed', 0)}/{ccounts.get('total', 0)})"
        )
    if any(chunk.get("status") == "completed" for chunk in chunks) and not BATCH_OUTPUT_PATH.is_file():
        print("→ Au moins un chunk terminé côté OpenAI — lancez download + ingest.")
    state
else:
    print("REFRESH_FROM_API=False — statut local uniquement")
"""
    ),
    py(
        """
# --- Fichiers locaux ---
from pathlib import Path

def count_jsonl_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

files = {
    "batch_state.json": BATCH_STATE_PATH.is_file(),
    "batch_output.jsonl": BATCH_OUTPUT_PATH.is_file(),
    "batch_errors.jsonl": BATCH_ERRORS_PATH.is_file(),
}
for name, exists in files.items():
    print(f"{name}: {'oui' if exists else 'non'}")

if BATCH_OUTPUT_PATH.is_file():
    print("Lignes batch_output :", count_jsonl_lines(BATCH_OUTPUT_PATH))
if BATCH_ERRORS_PATH.is_file():
    print("Lignes batch_errors :", count_jsonl_lines(BATCH_ERRORS_PATH))
"""
    ),
    py(
        """
# --- Download + ingest (autonome : ne dépend pas des cellules précédentes) ---
from annotation.batch_config import load_batch_config_from_run
from annotation.batch_client import download_batch_results, list_batch_chunks, refresh_batch_state
from annotation.batch_runner import ingest_batch_results

cfg = load_batch_config_from_run(RUN_ID, annotation_root=ANNOTATION_ROOT)

if not BATCH_OUTPUT_PATH.is_file():
    state = refresh_batch_state(cfg)
    completed = [
        chunk for chunk in list_batch_chunks(state)
        if chunk.get("status") == "completed"
    ]
    if not completed:
        raise RuntimeError(
            f"Aucun chunk terminé (status global={state.get('status')!r}). "
            "Attendez la fin du batch OpenAI."
        )
    dl = download_batch_results(cfg, partial=True)
    print("Téléchargé :", dl)
else:
    print("batch_output.jsonl déjà présent — download ignoré.")

df_final, meta = ingest_batch_results(cfg)
print(meta)
df_final.head()
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
