"""Libellés affichage notebook / figures (vocabulaire chaîne accidentelle, sans « macro »)."""

from __future__ import annotations

# Chaîne accidentelle A0 → A1 → B → C
CHAIN = "chaîne accidentelle"
CHAIN_STEPS = "étapes de la chaîne accidentelle"
CHAIN_LEGEND = "Étapes — chaîne accidentelle"
CHAIN_CAUSAL_STRUCTURE = "Structure causale — chaîne accidentelle"
CHAIN_TOPICS = "topics par étape"
CHAIN_MIN_STEPS = "étapes (≥ 2)"

STEP_PREDICTED = "Étape prédite (A0/A1/B/C)"
STEP_TRUE = "Étape réelle (A0/A1/B/C)"
STEP_CONFIDENCE = "Confiance étape (max p(m|u))"
STEP_DISTRIBUTION = "Répartition des étapes prédites"
STEP_DISTANCE_BOX = "Distances aux prototypes par étape prédite"

COL_CHAIN_PATH = "chaîne"
COL_STEP = "étape"

F1_STEPS = "F1 (étapes)"

# Renommage colonnes affichées (clé interne → libellé notebook)
DISPLAY_COLUMN_ALIASES: dict[str, str] = {
    "macro_path": COL_CHAIN_PATH,
    "pred_macro": STEP_PREDICTED,
    "true_macro": STEP_TRUE,
    "macro": COL_STEP,
    "dist_macro": "distance_étape",
}

__all__ = [
    "CHAIN",
    "CHAIN_STEPS",
    "CHAIN_LEGEND",
    "CHAIN_CAUSAL_STRUCTURE",
    "CHAIN_TOPICS",
    "CHAIN_MIN_STEPS",
    "STEP_PREDICTED",
    "STEP_TRUE",
    "STEP_CONFIDENCE",
    "STEP_DISTRIBUTION",
    "STEP_DISTANCE_BOX",
    "COL_CHAIN_PATH",
    "COL_STEP",
    "F1_STEPS",
    "DISPLAY_COLUMN_ALIASES",
    "rename_display_columns",
]


def rename_display_columns(df, aliases: dict[str, str] | None = None):
    """Renomme les colonnes pour affichage notebook (sans modifier les exports CSV)."""
    import pandas as pd

    mapping = aliases or DISPLAY_COLUMN_ALIASES
    use = {k: v for k, v in mapping.items() if k in df.columns}
    if not use:
        return df
    return df.rename(columns=use)
