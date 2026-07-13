"""Utilitaires pour l'annotation en deux passes (v13)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from annotation.cache import get_output_paths
from annotation.export_io import ANNOTATION_TABLE_SUFFIX
from annotation.prompts.v13_two_pass_ambiguity_context import should_run_second_pass

PASS1_OUTCOME_FIELDS = (
    "pred_injury_mentioned",
    "pred_hospitalized",
    "pred_fatal",
)

PASS1_ANNOTATION_FIELDS = (
    "label",
    "confidence",
    "ambiguous",
    "context_needed",
    "alternative_label",
    "ambiguity_type",
    "ambiguity_reason",
    "injury_mentioned",
    "hospitalized",
    "fatal",
)


def _row_cache_key(row: Mapping[str, Any]) -> str:
    accident_id = str(row.get("accident_id", "")).strip()
    fact_id = str(row.get("fact_id", "")).strip()
    if fact_id:
        return f"{accident_id}||{fact_id}"
    return accident_id


def build_first_pass_annotation(row: Mapping[str, Any]) -> dict[str, Any]:
    """Construit le bloc FIRST_PASS pour le prompt passe 2."""
    mapping = {
        "label": row.get("pred_label"),
        "confidence": row.get("pred_confidence"),
        "ambiguous": row.get("pred_ambiguous"),
        "context_needed": row.get("pred_context_needed"),
        "alternative_label": row.get("pred_alternative_label"),
        "ambiguity_type": row.get("pred_ambiguity_type"),
        "ambiguity_reason": row.get("pred_ambiguity_reason"),
        "injury_mentioned": row.get("pred_injury_mentioned"),
        "hospitalized": row.get("pred_hospitalized"),
        "fatal": row.get("pred_fatal"),
    }
    return {key: value for key, value in mapping.items() if value is not None and str(value) != ""}


def annotation_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Vue annotation (champs JSON) depuis une ligne annotée passe 1."""
    return {
        "label": row.get("pred_label"),
        "ambiguous": bool(row.get("pred_ambiguous", False)),
        "context_needed": bool(row.get("pred_context_needed", False)),
        "alternative_label": row.get("pred_alternative_label", "NONE"),
    }


def filter_pass2_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Sélectionne les unités devant être réexaminées en passe 2."""
    if df.empty:
        return df.copy()

    mask = df.apply(
        lambda row: should_run_second_pass(annotation_from_row(row)),
        axis=1,
    )
    return df.loc[mask].copy().reset_index(drop=True)


def find_pass1_annotated_xlsx(
    pass1_run_id: str,
    *,
    annotation_root: Path,
    openai_model: str,
    prompt_version: str,
    pass_mode: str = "pass1",
) -> Path:
    """Retourne le chemin du XLSX annoté de la passe 1."""
    from annotation.prompts import build_artifact_slug

    outputs_dir = annotation_root / "outputs" / pass1_run_id
    if not outputs_dir.is_dir():
        raise FileNotFoundError(f"Répertoire passe 1 introuvable : {outputs_dir}")

    artifact_slug = build_artifact_slug(prompt_version, pass_mode)
    _, _, annotated_path, _, _ = get_output_paths(
        outputs_dir,
        model_id=openai_model,
        prompt_version=prompt_version,
        artifact_slug=artifact_slug,
    )
    if annotated_path.is_file():
        return annotated_path

    matches = sorted(outputs_dir.glob(f"*__annotated{ANNOTATION_TABLE_SUFFIX}"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Aucun fichier *__annotated{ANNOTATION_TABLE_SUFFIX} dans {outputs_dir}"
        )
    raise FileNotFoundError(
        f"Plusieurs fichiers annotés dans {outputs_dir} ; précisez openai_model/prompt_version."
    )


def load_pass1_annotated(
    pass1_run_id: str,
    *,
    annotation_root: Path,
    openai_model: str,
    prompt_version: str,
    pass_mode: str = "pass1",
) -> pd.DataFrame:
    """Charge le tableau annoté complet de la passe 1."""
    path = find_pass1_annotated_xlsx(
        pass1_run_id,
        annotation_root=annotation_root,
        openai_model=openai_model,
        prompt_version=prompt_version,
        pass_mode=pass_mode,
    )
    return pd.read_excel(path, engine="openpyxl")


def _merge_key_columns(df: pd.DataFrame) -> list[str]:
    cols = ["accident_id", "fact_id"]
    if all(col in df.columns for col in cols):
        return cols
    if "accident_id" in df.columns:
        return ["accident_id"]
    raise ValueError("Colonnes accident_id/fact_id requises pour fusionner pass1 et pass2.")


def merge_pass1_pass2(
    pass1_df: pd.DataFrame,
    pass2_df: pd.DataFrame,
    *,
    key_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Fusionne les annotations : passe 2 remplace les lignes réannotées."""
    keys = key_cols or _merge_key_columns(pass1_df)
    out = pass1_df.copy().reset_index(drop=True)
    pass2_work = pass2_df.copy().reset_index(drop=True)

    if pass2_work.empty:
        out["pred_reannotated"] = False
        return out

    pass2_work["pred_reannotated"] = pass2_work.get("pred_ok", False).fillna(False).astype(bool)
    pass2_indexed = pass2_work.set_index(keys)

    for idx, row in out.iterrows():
        key = tuple(row[col] for col in keys)
        if key not in pass2_indexed.index:
            continue
        pass2_row = pass2_indexed.loc[key]
        if isinstance(pass2_row, pd.DataFrame):
            pass2_row = pass2_row.iloc[0]

        if not bool(pass2_row.get("pred_ok", False)):
            continue

        for col in pass2_row.index:
            if str(col).startswith("pred_"):
                out.at[idx, col] = pass2_row[col]

        for field in PASS1_OUTCOME_FIELDS:
            if field in row.index and pd.notna(row[field]):
                out.at[idx, field] = row[field]

        out.at[idx, "pred_reannotated"] = True

    if "pred_reannotated" not in out.columns:
        out["pred_reannotated"] = False
    else:
        out["pred_reannotated"] = (
            out["pred_reannotated"]
            .map(lambda v: bool(v) if pd.notna(v) else False)
        )

    return out


def pass2_selection_stats(pass1_df: pd.DataFrame) -> dict[str, Any]:
    """Statistiques de sélection pour la passe 2."""
    overview = pass2_ambiguity_overview(pass1_df)
    return overview["summary"]


def pass2_ambiguity_overview(pass1_df: pd.DataFrame) -> dict[str, Any]:
    """Vue d'ensemble des ambiguïtés passe 1 avant ré-annotation passe 2."""
    df = pass1_df.copy()
    total = int(len(df))
    empty = {
        "summary": {
            "n_pass1_units": 0,
            "n_pass2_candidates": 0,
            "pass2_candidate_rate": 0.0,
            "n_pred_ok": 0,
            "n_pred_not_ok": 0,
            "n_ambiguous": 0,
            "n_context_needed": 0,
            "n_alternative_label": 0,
            "n_accidents": 0,
            "n_accidents_with_candidate": 0,
        },
        "by_label": pd.DataFrame(),
        "by_ambiguity_type": pd.DataFrame(),
        "by_alternative_label": pd.DataFrame(),
        "candidates_preview": pd.DataFrame(),
    }
    if total == 0:
        return empty

    def _bool_series(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(False, index=df.index, dtype="boolean")
        return df[col].astype("boolean").fillna(False)

    pred_ok = _bool_series("pred_ok")
    ambiguous = _bool_series("pred_ambiguous")
    context_needed = _bool_series("pred_context_needed")
    if "pred_alternative_label" in df.columns:
        alternative = (
            df["pred_alternative_label"].fillna("NONE").astype(str).str.strip().str.upper() != "NONE"
        )
    else:
        alternative = pd.Series(False, index=df.index, dtype="boolean")

    candidate_mask = df.apply(
        lambda row: should_run_second_pass(annotation_from_row(row)),
        axis=1,
    )
    candidates = df.loc[candidate_mask].copy()
    n_candidates = int(len(candidates))

    n_accidents = int(df["accident_id"].nunique()) if "accident_id" in df.columns else 0
    n_accidents_with_candidate = (
        int(candidates["accident_id"].nunique()) if "accident_id" in candidates.columns else 0
    )

    summary = {
        "n_pass1_units": total,
        "n_pass2_candidates": n_candidates,
        "pass2_candidate_rate": (n_candidates / total) if total else 0.0,
        "n_pred_ok": int(pred_ok.sum()),
        "n_pred_not_ok": int((~pred_ok).sum()),
        "n_ambiguous": int(ambiguous.sum()),
        "n_context_needed": int(context_needed.sum()),
        "n_alternative_label": int(alternative.sum()),
        "n_accidents": n_accidents,
        "n_accidents_with_candidate": n_accidents_with_candidate,
        "pct_pass2_candidates": round(100.0 * n_candidates / total, 2) if total else 0.0,
    }

    by_label = pd.DataFrame()
    if n_candidates and "pred_label" in candidates.columns:
        by_label = (
            candidates["pred_label"]
            .fillna("MANQUANT")
            .astype(str)
            .value_counts()
            .rename_axis("pred_label")
            .reset_index(name="n_units")
        )

    by_ambiguity_type = pd.DataFrame()
    if n_candidates and "pred_ambiguity_type" in candidates.columns:
        by_ambiguity_type = (
            candidates["pred_ambiguity_type"]
            .fillna("MANQUANT")
            .astype(str)
            .value_counts()
            .rename_axis("pred_ambiguity_type")
            .reset_index(name="n_units")
        )

    by_alternative_label = pd.DataFrame()
    if n_candidates and "pred_alternative_label" in candidates.columns:
        by_alternative_label = (
            candidates["pred_alternative_label"]
            .fillna("NONE")
            .astype(str)
            .value_counts()
            .rename_axis("pred_alternative_label")
            .reset_index(name="n_units")
        )

    preview_cols = [
        col
        for col in (
            "accident_id",
            "fact_id",
            "sentence",
            "pred_label",
            "pred_alternative_label",
            "pred_ambiguity_type",
            "pred_ambiguous",
            "pred_context_needed",
        )
        if col in candidates.columns
    ]
    candidates_preview = candidates[preview_cols].head(10).reset_index(drop=True)

    return {
        "summary": summary,
        "by_label": by_label,
        "by_ambiguity_type": by_ambiguity_type,
        "by_alternative_label": by_alternative_label,
        "candidates_preview": candidates_preview,
    }
