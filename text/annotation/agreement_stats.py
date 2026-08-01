"""Accord inter-annotateurs (Observed agreement, Cohen's κ + IC 95 %)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from annotation.article_stats import find_annotated_xlsx
from annotation.prompts.v13_two_pass_ambiguity_context import LABELS

# Noms article (EN) pour le tableau d'accord — alignés sur le LaTeX cible.
AGREEMENT_CORPUS_DISPLAY: dict[str, str] = {
    "btp": "Construction",
    "metallurgie": "Metallurgy",
    "caou_chimie_plas": "Chemistry--plastics",
    "nicollin": "Company corpus",
}

_CORPUS_ORDER: tuple[str, ...] = (
    "btp",
    "caou_chimie_plas",
    "metallurgie",
    "nicollin",
)


def corpus_key_from_run_id(run_id: str) -> Optional[str]:
    """Extrait le suffixe corpus depuis ``run_all_X`` / ``run_partial_X``."""
    rid = str(run_id).strip()
    for prefix in ("run_partial_", "run_all_"):
        if rid.startswith(prefix):
            return rid[len(prefix) :]
    return None


def agreement_display_name(corpus_key: str) -> str:
    return AGREEMENT_CORPUS_DISPLAY.get(corpus_key, corpus_key.replace("_", " ").title())


def _valid_mask(df: pd.DataFrame) -> pd.Series:
    if "pred_ok" in df.columns:
        return df["pred_ok"].astype("boolean").fillna(False)
    if "pred_label" in df.columns:
        return df["pred_label"].astype(str).isin(LABELS)
    return pd.Series(False, index=df.index)


def pair_labels(
    df_all: pd.DataFrame,
    df_partial: pd.DataFrame,
    *,
    label_col: str = "pred_label",
    accident_col: str = "accident_id",
    fact_col: str = "fact_id",
) -> pd.DataFrame:
    """Jointure ``(accident_id, fact_id)`` avec labels valides des deux côtés."""
    for name, df in (("all", df_all), ("partial", df_partial)):
        missing = {accident_col, fact_col, label_col} - set(df.columns)
        if missing:
            raise KeyError(f"Colonnes manquantes dans {name} : {sorted(missing)}")

    extra_all = [
        c
        for c in (
            "sentence",
            "accident_summary",
            "summary_accident",
            "pred_justification",
        )
        if c in df_all.columns and c not in {accident_col, fact_col, label_col}
    ]
    left_cols = [accident_col, fact_col, label_col, *extra_all]
    left = df_all.loc[_valid_mask(df_all), left_cols].copy()
    right = df_partial.loc[_valid_mask(df_partial), [accident_col, fact_col, label_col]].copy()
    left[accident_col] = left[accident_col].astype(str)
    right[accident_col] = right[accident_col].astype(str)
    left[fact_col] = left[fact_col].astype(str)
    right[fact_col] = right[fact_col].astype(str)
    left[label_col] = left[label_col].astype(str)
    right[label_col] = right[label_col].astype(str)
    rename_left = {label_col: "label_all"}
    if "pred_justification" in left.columns:
        rename_left["pred_justification"] = "justification_all"
    left = left.rename(columns=rename_left)
    right = right.rename(columns={label_col: "label_partial"})
    # Justification Annotator 2 si disponible.
    if "pred_justification" in df_partial.columns:
        right_j = df_partial.loc[
            _valid_mask(df_partial), [accident_col, fact_col, "pred_justification"]
        ].copy()
        right_j[accident_col] = right_j[accident_col].astype(str)
        right_j[fact_col] = right_j[fact_col].astype(str)
        right_j = right_j.rename(columns={"pred_justification": "justification_partial"})
        right = right.merge(right_j, on=[accident_col, fact_col], how="left")

    merged = left.merge(right, on=[accident_col, fact_col], how="inner")
    merged = merged[
        merged["label_all"].isin(LABELS) & merged["label_partial"].isin(LABELS)
    ].reset_index(drop=True)
    return merged


def disagreement_subset(
    paired: pd.DataFrame,
    *,
    labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Unités où les deux annotateurs divergent.

    Si ``labels`` est fourni, ne garde que les désaccords où les deux labels
    appartiennent à cet ensemble (ex. ``("B", "C")``). Sinon, tous les
    désaccords ``label_all != label_partial``.
    """
    if paired.empty:
        return paired.iloc[0:0].copy()
    y1 = paired["label_all"].astype(str)
    y2 = paired["label_partial"].astype(str)
    mask = y1 != y2
    if labels is not None:
        labs = {str(lab) for lab in labels}
        mask = mask & y1.isin(labs) & y2.isin(labs)
    out = paired.loc[mask].copy().reset_index(drop=True)
    if out.empty:
        return out
    out["disagreement"] = (
        out["label_all"].astype(str) + " → " + out["label_partial"].astype(str)
    )
    return out


def role_counts_in_sample(
    paired: pd.DataFrame,
    *,
    label_col: str = "label_all",
    labels: Sequence[str] = LABELS,
) -> dict[str, int]:
    """Effectifs par rôle dans l'échantillon apparié (référence ``label_all`` = run_all)."""
    if paired.empty or label_col not in paired.columns:
        return {f"n_{lab}": 0 for lab in labels}
    counts = paired[label_col].astype(str).value_counts()
    return {f"n_{lab}": int(counts.get(lab, 0)) for lab in labels}


def confusion_matrix_from_paired(
    paired: pd.DataFrame,
    *,
    labels: Sequence[str] = LABELS,
    row_annotator: str = "run_all",
    col_annotator: str = "run_partial",
) -> pd.DataFrame:
    """Matrice de confusion : lignes = run_all, colonnes = run_partial."""
    label_list = [str(lab) for lab in labels]
    if paired.empty:
        cm = np.zeros((len(label_list), len(label_list)), dtype=int)
    else:
        cm = confusion_matrix(
            paired["label_all"].astype(str),
            paired["label_partial"].astype(str),
            labels=label_list,
        )
    out = pd.DataFrame(cm, index=label_list, columns=label_list)
    out.index.name = row_annotator
    out.columns.name = col_annotator
    return out


def observed_agreement(y1: Sequence[str], y2: Sequence[str]) -> float:
    a = np.asarray(y1, dtype=object)
    b = np.asarray(y2, dtype=object)
    if len(a) == 0:
        return float("nan")
    return float(np.mean(a == b))


def cohen_kappa(y1: Sequence[str], y2: Sequence[str]) -> float:
    a = np.asarray(y1, dtype=object)
    b = np.asarray(y2, dtype=object)
    if len(a) == 0:
        return float("nan")
    return float(cohen_kappa_score(a, b))


def bootstrap_kappa_ci(
    y1: Sequence[str],
    y2: Sequence[str],
    *,
    accident_ids: Sequence[str],
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """IC 95 % bootstrap non paramétrique sur Cohen's κ (niveau récit)."""
    a = np.asarray(y1, dtype=object)
    b = np.asarray(y2, dtype=object)
    ids = np.asarray(accident_ids, dtype=object)
    if len(a) == 0:
        return float("nan"), float("nan")
    if len(a) != len(b) or len(a) != len(ids):
        raise ValueError(
            "y1, y2 et accident_ids doivent avoir la même longueur "
            f"({len(a)}, {len(b)}, {len(ids)})."
        )

    unique_ids = np.unique(ids)
    n_narratives = len(unique_ids)
    if n_narratives == 0:
        return float("nan"), float("nan")

    indices_by_narrative = {
        uid: np.flatnonzero(ids == uid) for uid in unique_ids
    }

    rng = np.random.RandomState(int(seed))
    scores = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sampled_ids = rng.choice(unique_ids, size=n_narratives, replace=True)
        idx = np.concatenate([indices_by_narrative[sid] for sid in sampled_ids])
        scores[i] = cohen_kappa_score(a[idx], b[idx])
    lo = float(np.nanpercentile(scores, 100.0 * (alpha / 2.0)))
    hi = float(np.nanpercentile(scores, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def agreement_metrics_from_paired(
    paired: pd.DataFrame,
    *,
    corpus: str,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    empty_roles = {f"n_{lab}": 0 for lab in LABELS}
    if paired.empty:
        return {
            "corpus": corpus,
            "n_narratives": 0,
            "n_factual_units": 0,
            **empty_roles,
            "observed_agreement": float("nan"),
            "observed_agreement_pct": float("nan"),
            "kappa": float("nan"),
            "kappa_ci_low": float("nan"),
            "kappa_ci_high": float("nan"),
            "kappa_display": "—",
        }
    y1 = paired["label_all"].astype(str).tolist()
    y2 = paired["label_partial"].astype(str).tolist()
    oa = observed_agreement(y1, y2)
    kappa = cohen_kappa(y1, y2)
    if "accident_id" not in paired.columns:
        raise KeyError("accident_id requis pour le bootstrap au niveau récit.")
    accident_ids = paired["accident_id"].astype(str).tolist()
    lo, hi = bootstrap_kappa_ci(
        y1, y2, accident_ids=accident_ids, n_boot=n_boot, seed=seed
    )
    n_narr = (
        int(paired["accident_id"].nunique())
        if "accident_id" in paired.columns
        else 0
    )
    return {
        "corpus": corpus,
        "n_narratives": n_narr,
        "n_factual_units": int(len(paired)),
        **role_counts_in_sample(paired),
        "observed_agreement": oa,
        "observed_agreement_pct": round(100.0 * oa, 1) if np.isfinite(oa) else float("nan"),
        "kappa": kappa,
        "kappa_ci_low": lo,
        "kappa_ci_high": hi,
        "kappa_display": (
            f"{kappa:.2f} [{lo:.2f}, {hi:.2f}]"
            if np.isfinite(kappa) and np.isfinite(lo) and np.isfinite(hi)
            else "—"
        ),
    }


def discover_agreement_pairs(
    outputs_dir: Path,
) -> list[dict[str, Any]]:
    """
    Apparie ``run_partial_X`` avec ``run_all_X`` lorsqu'un ``*__annotated.xlsx``
    est présent des deux côtés.
    """
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.is_dir():
        return []

    all_by_key: dict[str, Path] = {}
    partial_by_key: dict[str, Path] = {}
    for run_dir in sorted(outputs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        annotated = find_annotated_xlsx(run_dir)
        if annotated is None:
            continue
        run_id = run_dir.name
        key = corpus_key_from_run_id(run_id)
        if key is None:
            continue
        if run_id.startswith("run_all_"):
            all_by_key[key] = annotated
        elif run_id.startswith("run_partial_"):
            partial_by_key[key] = annotated

    pairs: list[dict[str, Any]] = []
    for key in _CORPUS_ORDER:
        if key in all_by_key and key in partial_by_key:
            pairs.append(
                {
                    "corpus_key": key,
                    "corpus": agreement_display_name(key),
                    "all_path": all_by_key[key],
                    "partial_path": partial_by_key[key],
                    "run_all_id": f"run_all_{key}",
                    "run_partial_id": f"run_partial_{key}",
                }
            )
    for key in sorted(set(all_by_key) & set(partial_by_key)):
        if key in _CORPUS_ORDER:
            continue
        pairs.append(
            {
                "corpus_key": key,
                "corpus": agreement_display_name(key),
                "all_path": all_by_key[key],
                "partial_path": partial_by_key[key],
                "run_all_id": f"run_all_{key}",
                "run_partial_id": f"run_partial_{key}",
            }
        )
    return pairs


def build_agreement_artifacts(
    pairs: Sequence[Mapping[str, Any]],
    *,
    n_boot: int = 2000,
    seed: int = 42,
    include_overall: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Tableau d'accord + matrices de confusion + paires annotées par corpus."""
    rows: list[dict[str, Any]] = []
    paired_frames: list[pd.DataFrame] = []
    confusion: dict[str, pd.DataFrame] = {}
    paired_by_corpus: dict[str, pd.DataFrame] = {}

    for pair in pairs:
        df_all = pd.read_excel(pair["all_path"], engine="openpyxl")
        df_partial = pd.read_excel(pair["partial_path"], engine="openpyxl")
        paired = pair_labels(df_all, df_partial)
        paired = paired.copy()
        paired.insert(0, "corpus", str(pair["corpus"]))
        paired_frames.append(paired)
        corpus_name = str(pair["corpus"])
        paired_by_corpus[corpus_name] = paired
        rows.append(
            agreement_metrics_from_paired(
                paired,
                corpus=corpus_name,
                n_boot=n_boot,
                seed=seed,
            )
        )
        confusion[corpus_name] = confusion_matrix_from_paired(paired)

    if include_overall and paired_frames:
        overall = pd.concat(paired_frames, ignore_index=True)
        rows.append(
            agreement_metrics_from_paired(
                overall,
                corpus="Overall",
                n_boot=n_boot,
                seed=seed,
            )
        )
        confusion["Overall"] = confusion_matrix_from_paired(overall)
        paired_by_corpus["Overall"] = overall

    role_cols = [f"n_{lab}" for lab in LABELS]
    empty_cols = [
        "corpus",
        "n_narratives",
        "n_factual_units",
        *role_cols,
        "observed_agreement_pct",
        "kappa",
        "kappa_ci_low",
        "kappa_ci_high",
        "kappa_display",
    ]
    if not rows:
        return pd.DataFrame(columns=empty_cols), {}, {}
    return pd.DataFrame(rows), confusion, paired_by_corpus


def build_agreement_table(
    pairs: Sequence[Mapping[str, Any]],
    *,
    n_boot: int = 2000,
    seed: int = 42,
    include_overall: bool = True,
) -> pd.DataFrame:
    """Tableau article : OA + κ (IC 95 %) + effectifs par rôle (+ Overall)."""
    table, _, _ = build_agreement_artifacts(
        pairs, n_boot=n_boot, seed=seed, include_overall=include_overall
    )
    return table


def export_agreement_latex(df: pd.DataFrame) -> str:
    """Export LaTeX ``table*`` / ``tabularx`` (format article)."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        "",
        r"\caption{Inter-annotator agreement for accident-process role annotation.",
        r"Role counts ($n_{\mathrm{A0}}$--$n_{\mathrm{C}}$) are taken from the",
        r"\texttt{run\_all} labels on the paired sample.",
        r"Statistics were calculated from independently assigned labels before",
        r"adjudication. Ninety-five percent confidence intervals for Cohen's",
        r"$\kappa$ were obtained by cluster bootstrap at the narrative level.}",
        r"\label{tab:annotation_agreement}",
        "",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{3.5pt}",
        "",
        r"\begin{tabularx}{\textwidth}{",
        r"    >{\raggedright\arraybackslash}X",
        r"    >{\centering\arraybackslash}c",
        r"    >{\centering\arraybackslash}c",
        r"    >{\centering\arraybackslash}c",
        r"    >{\centering\arraybackslash}c",
        r"    >{\centering\arraybackslash}c",
        r"    >{\centering\arraybackslash}c",
        r"    >{\centering\arraybackslash}c",
        r"    >{\centering\arraybackslash}c",
        r"}",
        "",
        r"\toprule",
        r"\textbf{Corpus} &",
        r"\textbf{Narr.} &",
        r"\textbf{Units} &",
        r"\textbf{$n_{\mathrm{A0}}$} &",
        r"\textbf{$n_{\mathrm{A1}}$} &",
        r"\textbf{$n_{\mathrm{B}}$} &",
        r"\textbf{$n_{\mathrm{C}}$} &",
        r"\textbf{Obs.\ agr.} &",
        r"\textbf{Cohen's $\kappa$ (95\% CI)} \\",
        r"\midrule",
        "",
    ]

    def _fmt_int(x: Any) -> str:
        try:
            return f"{int(x):,}"
        except (TypeError, ValueError):
            return "—"

    def _fmt_pct(x: Any) -> str:
        try:
            v = float(x)
            if not np.isfinite(v):
                return "—"
            return f"{v:.1f}\\%"
        except (TypeError, ValueError):
            return "—"

    def _row_tex(corpus: str, row: pd.Series) -> str:
        return (
            f"{corpus} &\n"
            f"{_fmt_int(row.get('n_narratives'))} &\n"
            f"{_fmt_int(row.get('n_factual_units'))} &\n"
            f"{_fmt_int(row.get('n_A0'))} &\n"
            f"{_fmt_int(row.get('n_A1'))} &\n"
            f"{_fmt_int(row.get('n_B'))} &\n"
            f"{_fmt_int(row.get('n_C'))} &\n"
            f"{_fmt_pct(row.get('observed_agreement_pct'))} &\n"
            f"{row.get('kappa_display', '—')} \\\\"
        )

    body_rows = df[df["corpus"].astype(str) != "Overall"] if "corpus" in df.columns else df
    overall_rows = (
        df[df["corpus"].astype(str) == "Overall"] if "corpus" in df.columns else df.iloc[0:0]
    )

    for _, row in body_rows.iterrows():
        lines.append(_row_tex(str(row.get("corpus", "")), row))
        lines.append("")

    if not overall_rows.empty:
        lines.append(r"\midrule")
        lines.append("")
        for _, row in overall_rows.iterrows():
            lines.append(_row_tex("Overall", row))
            lines.append("")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def export_confusion_latex(
    cm: pd.DataFrame,
    *,
    corpus: str,
    caption: Optional[str] = None,
) -> str:
    """Petite table LaTeX pour une matrice de confusion."""
    labels = [str(c) for c in cm.columns]
    cap = caption or (
        f"Confusion matrix for {corpus}: rows = Annotator 1, "
        r"columns = Annotator 2."
    )
    label_slug = "".join(ch if ch.isalnum() else "_" for ch in corpus.lower())
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        f"\\caption{{{cap}}}",
        f"\\label{{tab:confusion_{label_slug}}}",
        r"\begin{tabular}{l" + "c" * len(labels) + "}",
        r"\toprule",
        "Annotator 1 $\\backslash$ Annotator 2 & " + " & ".join(labels) + r" \\",
        r"\midrule",
    ]
    for lab in labels:
        vals = " & ".join(str(int(cm.loc[lab, c])) for c in labels)
        lines.append(f"{lab} & {vals} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "AGREEMENT_CORPUS_DISPLAY",
    "agreement_display_name",
    "agreement_metrics_from_paired",
    "bootstrap_kappa_ci",
    "build_agreement_artifacts",
    "build_agreement_table",
    "cohen_kappa",
    "confusion_matrix_from_paired",
    "corpus_key_from_run_id",
    "disagreement_subset",
    "discover_agreement_pairs",
    "export_agreement_latex",
    "export_confusion_latex",
    "observed_agreement",
    "pair_labels",
    "role_counts_in_sample",
]
