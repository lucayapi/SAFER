"""Statistiques d'annotation multi-corpus pour article / exploration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from annotation.export_io import ANNOTATION_TABLE_SUFFIX
from annotation.prompts.v13_two_pass_ambiguity_context import LABELS

# Palette article (Okabe–Ito, lisible N&B) — barplots et heatmaps de confusion.
ARTICLE_PLOT_COLORS: tuple[str, ...] = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
)

# Types d'ambiguïté correspondant à des unités multi-fonctions (mixtes).
MIXED_AMBIGUITY_TYPES: tuple[str, ...] = (
    "A0_A1",
    "A1_B",
    "B_C",
    "MULTIPLE_ROLES",
)
MIXED_TYPE_DISPLAY: dict[str, str] = {
    "A0_A1": "A0/A1",
    "A1_B": "A1/B",
    "B_C": "B/C",
    "MULTIPLE_ROLES": "MULTIPLE_ROLES",
}

_DEFAULT_CORPUS_NAMES: dict[str, str] = {
    "run_all_btp": "BTP",
    "run_all_metallurgie": "Métallurgie",
    "run_all_caou_chimie_plas": "Caoutchouc-chimie-plastiques",
    "run_all_nicollin": "Company corpus",
    "run_all_bois_textille": "Bois-textile",
    "run_all_service_temporaire": "Service temporaire",
}

_DATASET_CORPUS_ORDER: tuple[str, ...] = (
    "btp",
    "caou",
    "metallurgie",
    "nicollin",
)

# Noms EN alignés sur le tableau IAA (agreement_stats.AGREEMENT_CORPUS_DISPLAY).
_DATASET_CORPUS_DISPLAY_EN: dict[str, str] = {
    "btp": "Construction",
    "metallurgie": "Metallurgy",
    "caou": "Chemistry--plastics",
    "nicollin": "Company corpus",
}

_ANNOTATION_LOAD_COLS: tuple[str, ...] = (
    "accident_id",
    "fact_id",
    "pred_label",
    "pred_ok",
    "pred_ambiguity_type",
)


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def corpus_display_name(run_id: str, run_config: Optional[dict[str, Any]] = None) -> str:
    """Nom lisible du corpus pour tableaux / graphiques."""
    if run_id in _DEFAULT_CORPUS_NAMES:
        return _DEFAULT_CORPUS_NAMES[run_id]

    cfg = run_config or {}
    input_csv = str(cfg.get("input_csv") or "").strip()
    if input_csv:
        stem = Path(input_csv).stem
        return stem.replace("_sentence_accidents", "").replace("_", " ").strip().title()

    basename = str(cfg.get("output_basename") or "").strip()
    if basename:
        return basename.replace("_v13_pass1", "").replace("_", " ").strip().title()

    return run_id


def find_annotated_xlsx(run_dir: Path) -> Optional[Path]:
    """Cherche le fichier *__annotated.xlsx dans un dossier de run."""
    matches = sorted(run_dir.glob(f"*__annotated{ANNOTATION_TABLE_SUFFIX}"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return None
    # Préfère le fichier le plus récent si plusieurs.
    return max(matches, key=lambda path: path.stat().st_mtime)


def article_plot_colors(n: int) -> list[str]:
    """Couleurs cyclées depuis ``ARTICLE_PLOT_COLORS``."""
    if n <= 0:
        return []
    palette = list(ARTICLE_PLOT_COLORS)
    return [palette[i % len(palette)] for i in range(n)]


def article_confusion_cmap():
    """Colormap confusion alignée sur la palette article (blanc → bleus/verts)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "article_iaa",
        ["#ffffff", ARTICLE_PLOT_COLORS[4], ARTICLE_PLOT_COLORS[0], ARTICLE_PLOT_COLORS[2]],
    )


def read_annotation_table(path: Path, *, usecols: Optional[list[str]] = None) -> pd.DataFrame:
    """Charge un export annoté CSV (dataset/) ou XLSX (annotation/outputs)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        peek = pd.read_csv(path, nrows=0)
        cols = None
        if usecols is not None:
            cols = [c for c in usecols if c in peek.columns]
        elif any(c in peek.columns for c in _ANNOTATION_LOAD_COLS):
            cols = [c for c in _ANNOTATION_LOAD_COLS if c in peek.columns]
        return pd.read_csv(path, usecols=cols)
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(path, engine="openpyxl")
    raise ValueError(f"Format non supporté pour annotation : {path}")


def discover_dataset_corpora(dataset_dir: Path) -> list[dict[str, Any]]:
    """Liste ``dataset/data_<id>.csv`` / ``.xlsx`` présents, ordre article fixe.

    Si CSV et XLSX coexistent pour le même id, le CSV est préféré.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        return []

    found: dict[str, Path] = {}
    # XLSX d'abord, puis CSV pour écraser (préférence CSV).
    for pattern in ("data_*.xlsx", "data_*.xlsm", "data_*.csv"):
        for path in sorted(dataset_dir.glob(pattern)):
            corpus_id = path.stem[len("data_") :]
            if corpus_id:
                found[corpus_id] = path

    ordered_ids = [cid for cid in _DATASET_CORPUS_ORDER if cid in found]
    ordered_ids.extend(sorted(cid for cid in found if cid not in _DATASET_CORPUS_ORDER))

    corpora: list[dict[str, Any]] = []
    for corpus_id in ordered_ids:
        corpora.append(
            {
                "corpus_id": corpus_id,
                "run_id": f"dataset_{corpus_id}",
                "corpus": _DATASET_CORPUS_DISPLAY_EN.get(
                    corpus_id, corpus_id.replace("_", " ").title()
                ),
                "annotated_path": found[corpus_id],
            }
        )
    return corpora


def discover_annotation_runs(
    outputs_dir: Path,
    *,
    full_runs_only: bool = False,
) -> list[dict[str, Any]]:
    """Liste les runs avec un export annoté disponible.

    Si ``full_runs_only=True``, ne garde que les ``run_all_*`` (exclut les
    ``run_partial_*`` utilisés pour l'accord IAA).
    """
    if not outputs_dir.is_dir():
        return []

    runs: list[dict[str, Any]] = []
    for run_dir in sorted(outputs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        annotated_path = find_annotated_xlsx(run_dir)
        if annotated_path is None:
            continue
        run_config = _load_run_config(run_dir)
        run_id = str(run_config.get("run_id") or run_dir.name)
        if full_runs_only and not str(run_id).startswith("run_all_"):
            continue
        runs.append(
            {
                "run_id": run_id,
                "corpus": corpus_display_name(run_id, run_config),
                "run_dir": run_dir,
                "annotated_path": annotated_path,
                "run_config": run_config,
            }
        )
    return runs


def _valid_annotation_mask(df: pd.DataFrame) -> pd.Series:
    if "pred_ok" in df.columns:
        return df["pred_ok"].astype("boolean").fillna(False)
    if "pred_label" in df.columns:
        return df["pred_label"].isin(LABELS)
    return pd.Series(False, index=df.index)


def corpus_annotation_summary(df: pd.DataFrame, *, corpus: str) -> dict[str, Any]:
    """Résumé d'un corpus annoté.

    ``n_recits`` = récits avec **au moins une** unité ``pred_ok`` (aligné sur
    le dénombrement IAA / analyses de labels). Les récits dont toutes les
    unités ont échoué la validation restent dans ``n_units`` via
    ``n_missing_or_failed``.
    """
    n_units = int(len(df))
    valid = df.loc[_valid_annotation_mask(df)].copy()
    n_annotated = int(len(valid))
    n_recits = (
        int(valid["accident_id"].nunique()) if "accident_id" in valid.columns else 0
    )

    label_counts = (
        valid["pred_label"].value_counts()
        if "pred_label" in valid.columns
        else pd.Series(dtype=int)
    )
    denom = int(label_counts.sum()) if len(label_counts) else 0

    pct: dict[str, float] = {}
    for label in LABELS:
        count = int(label_counts.get(label, 0))
        pct[f"pct_{label}"] = round(100.0 * count / denom, 2) if denom else 0.0
        pct[f"n_{label}"] = count

    return {
        "corpus": corpus,
        "n_recits": n_recits,
        "n_units": n_units,
        "n_annotated": n_annotated,
        "n_missing_or_failed": n_units - n_annotated,
        **pct,
    }


def build_summary_table(
    runs: list[dict[str, Any]],
    *,
    include_overall: bool = True,
) -> pd.DataFrame:
    """Tableau article : récits, unités, % par classe (+ Overall optionnel)."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        df = read_annotation_table(Path(run["annotated_path"]))
        rows.append(corpus_annotation_summary(df, corpus=str(run["corpus"])))
    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)

    if include_overall and len(table) > 0:
        overall: dict[str, Any] = {"corpus": "Overall"}
        for col in (
            "n_recits",
            "n_units",
            "n_annotated",
            "n_missing_or_failed",
            "n_A0",
            "n_A1",
            "n_B",
            "n_C",
        ):
            if col in table.columns:
                overall[col] = int(table[col].fillna(0).sum())
        denom = int(overall.get("n_annotated", 0) or 0)
        for label in LABELS:
            n_key = f"n_{label}"
            count = int(overall.get(n_key, 0) or 0)
            overall[f"pct_{label}"] = round(100.0 * count / denom, 2) if denom else 0.0
        table = pd.concat([table, pd.DataFrame([overall])], ignore_index=True)

    ordered_cols = [
        "corpus",
        "n_recits",
        "n_units",
        "n_annotated",
        "n_missing_or_failed",
        "pct_A0",
        "pct_A1",
        "pct_B",
        "pct_C",
        "n_A0",
        "n_A1",
        "n_B",
        "n_C",
    ]
    existing = [col for col in ordered_cols if col in table.columns]
    return table[existing]


def build_label_distribution_long(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Format long pour graphiques (corpus × label × effectif / %)."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        df = read_annotation_table(Path(run["annotated_path"]))
        valid = df.loc[_valid_annotation_mask(df)]
        counts = valid["pred_label"].value_counts() if "pred_label" in valid.columns else pd.Series(dtype=int)
        total = int(counts.sum())
        for label in LABELS:
            count = int(counts.get(label, 0))
            rows.append(
                {
                    "corpus": str(run["corpus"]),
                    "run_id": str(run["run_id"]),
                    "label": label,
                    "n_units": count,
                    "pct": round(100.0 * count / total, 2) if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _normalize_ambiguity_type(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def mixed_units_mask(df: pd.DataFrame) -> pd.Series:
    """Masque des unités annotées OK avec un type d'ambiguïté multi-fonctions."""
    valid = _valid_annotation_mask(df)
    if "pred_ambiguity_type" not in df.columns:
        return pd.Series(False, index=df.index)
    amb = _normalize_ambiguity_type(df["pred_ambiguity_type"])
    return valid & amb.isin(MIXED_AMBIGUITY_TYPES)


def mixed_units_summary(df: pd.DataFrame, *, corpus: str) -> dict[str, Any]:
    """Résumé des unités mixtes pour un corpus."""
    valid = df.loc[_valid_annotation_mask(df)]
    n_annotated = int(len(valid))
    mixed = df.loc[mixed_units_mask(df)]
    n_mixed = int(len(mixed))

    by_type: dict[str, int] = {}
    if n_mixed and "pred_ambiguity_type" in mixed.columns:
        counts = _normalize_ambiguity_type(mixed["pred_ambiguity_type"]).value_counts()
        for raw_type in MIXED_AMBIGUITY_TYPES:
            by_type[f"n_{raw_type}"] = int(counts.get(raw_type, 0))
    else:
        for raw_type in MIXED_AMBIGUITY_TYPES:
            by_type[f"n_{raw_type}"] = 0

    return {
        "corpus": corpus,
        "n_annotated": n_annotated,
        "n_mixed": n_mixed,
        "pct_mixed": round(100.0 * n_mixed / n_annotated, 2) if n_annotated else 0.0,
        **by_type,
    }


def build_mixed_units_table(
    runs: list[dict[str, Any]],
    *,
    include_overall: bool = True,
) -> pd.DataFrame:
    """Tableau : proportion d'unités mixtes par corpus (+ Overall optionnel)."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        df = read_annotation_table(Path(run["annotated_path"]))
        rows.append(mixed_units_summary(df, corpus=str(run["corpus"])))
    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)

    if include_overall and len(table) > 0:
        overall: dict[str, Any] = {"corpus": "Overall"}
        for col in ["n_annotated", "n_mixed", *[f"n_{t}" for t in MIXED_AMBIGUITY_TYPES]]:
            if col in table.columns:
                overall[col] = int(table[col].fillna(0).sum())
        denom = int(overall.get("n_annotated", 0) or 0)
        n_mixed = int(overall.get("n_mixed", 0) or 0)
        overall["pct_mixed"] = round(100.0 * n_mixed / denom, 2) if denom else 0.0
        table = pd.concat([table, pd.DataFrame([overall])], ignore_index=True)

    ordered = [
        "corpus",
        "n_annotated",
        "n_mixed",
        "pct_mixed",
        *[f"n_{t}" for t in MIXED_AMBIGUITY_TYPES],
    ]
    return table[[c for c in ordered if c in table.columns]]


def build_mixed_combination_long(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Format long : corpus × combinaison (A0/A1, A1/B, …) × n / % parmi mixtes."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        df = read_annotation_table(Path(run["annotated_path"]))
        mixed = df.loc[mixed_units_mask(df)]
        n_mixed = int(len(mixed))
        if n_mixed and "pred_ambiguity_type" in mixed.columns:
            counts = _normalize_ambiguity_type(mixed["pred_ambiguity_type"]).value_counts()
        else:
            counts = pd.Series(dtype=int)

        for raw_type in MIXED_AMBIGUITY_TYPES:
            count = int(counts.get(raw_type, 0))
            rows.append(
                {
                    "corpus": str(run["corpus"]),
                    "run_id": str(run["run_id"]),
                    "ambiguity_type": raw_type,
                    "combination": MIXED_TYPE_DISPLAY[raw_type],
                    "n_units": count,
                    "pct_of_mixed": round(100.0 * count / n_mixed, 2) if n_mixed else 0.0,
                    "n_mixed": n_mixed,
                }
            )
    return pd.DataFrame(rows)


def build_label_distribution_sensitivity(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """% labels avec toutes les unités vs après exclusion des mixtes (par corpus)."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        df = read_annotation_table(Path(run["annotated_path"]))
        valid = df.loc[_valid_annotation_mask(df)]
        mixed_mask = mixed_units_mask(df)
        excl = df.loc[_valid_annotation_mask(df) & ~mixed_mask]

        counts_all = (
            valid["pred_label"].value_counts() if "pred_label" in valid.columns else pd.Series(dtype=int)
        )
        counts_excl = (
            excl["pred_label"].value_counts() if "pred_label" in excl.columns else pd.Series(dtype=int)
        )
        n_all = int(counts_all.sum())
        n_excl = int(counts_excl.sum())
        n_excluded = int(mixed_mask.sum())

        row: dict[str, Any] = {
            "corpus": str(run["corpus"]),
            "run_id": str(run["run_id"]),
            "n_annotated": n_all,
            "n_excluded_mixed": n_excluded,
            "n_after_exclusion": n_excl,
        }
        for label in LABELS:
            n_a = int(counts_all.get(label, 0))
            n_e = int(counts_excl.get(label, 0))
            row[f"n_{label}_all"] = n_a
            row[f"pct_{label}_all"] = round(100.0 * n_a / n_all, 2) if n_all else 0.0
            row[f"n_{label}_excl_mixed"] = n_e
            row[f"pct_{label}_excl_mixed"] = round(100.0 * n_e / n_excl, 2) if n_excl else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_label_distribution_sensitivity_long(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Format long pour graphique de sensibilité (corpus × label × condition)."""
    wide = build_label_distribution_sensitivity(runs)
    if wide.empty:
        return pd.DataFrame(
            columns=["corpus", "run_id", "label", "condition", "n_units", "pct"]
        )

    rows: list[dict[str, Any]] = []
    for _, row in wide.iterrows():
        for label in LABELS:
            rows.append(
                {
                    "corpus": row["corpus"],
                    "run_id": row["run_id"],
                    "label": label,
                    "condition": "all",
                    "n_units": int(row[f"n_{label}_all"]),
                    "pct": float(row[f"pct_{label}_all"]),
                }
            )
            rows.append(
                {
                    "corpus": row["corpus"],
                    "run_id": row["run_id"],
                    "label": label,
                    "condition": "excl_mixed",
                    "n_units": int(row[f"n_{label}_excl_mixed"]),
                    "pct": float(row[f"pct_{label}_excl_mixed"]),
                }
            )
    return pd.DataFrame(rows)
