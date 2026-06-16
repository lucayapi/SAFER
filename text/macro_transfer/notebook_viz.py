"""Visualisations notebook pour macro_transfer (UMAP, DataMapPlot, PCA/t-SNE)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from macro_transfer.constants import MACRO_NAMES
from safer_core.display_labels import (
    CHAIN,
    CHAIN_LEGEND,
    CHAIN_TOPICS,
    F1_STEPS,
    STEP_CONFIDENCE,
    STEP_DISTRIBUTION,
    STEP_DISTANCE_BOX,
    STEP_PREDICTED,
    STEP_TRUE,
)

# Couleurs fixes par macro (lisibles sur fond pastel).
MACRO_COLOR_HEX: Dict[str, str] = {
    "A0": "#1f77b4",
    "A1": "#ff7f0e",
    "B": "#2ca02c",
    "C": "#d62728",
}


@dataclass
class FSPRunArtifacts:
    """Artefacts d'un run Frozen Source Prototypes."""

    out_dir: Path
    transfer_dir: Path
    predictions: pd.DataFrame
    prototypes: pd.DataFrame
    metrics: Dict[str, Any]
    confusion: Optional[pd.DataFrame]
    classification_report: Optional[pd.DataFrame]


def _read_optional_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    return pd.read_csv(path)


def load_fsp_run_artifacts(out_dir: str | Path) -> FSPRunArtifacts:
    """Charge les sorties baseline Frozen Source Prototypes."""
    root = Path(out_dir).resolve()
    transfer = root / "transfer"
    if not transfer.is_dir():
        # Compat: certains runs FSP sont stockés sous .../frozen_source_prototypes/{scgm,raw}/transfer
        for variant in ("scgm", "raw"):
            cand = root / variant / "transfer"
            if cand.is_dir():
                root = root / variant
                transfer = cand
                break
    preds_path = transfer / "target_macro_predictions.csv"
    protos_path = transfer / "source_prototypes.csv"
    if not preds_path.is_file():
        raise FileNotFoundError(f"Prédictions FSP manquantes : {preds_path}")
    if not protos_path.is_file():
        raise FileNotFoundError(f"Prototypes FSP manquants : {protos_path}")
    metrics: Dict[str, Any] = {}
    metrics_path = transfer / "metrics.json"
    if metrics_path.is_file():
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
    return FSPRunArtifacts(
        out_dir=root,
        transfer_dir=transfer,
        predictions=pd.read_csv(preds_path),
        prototypes=pd.read_csv(protos_path),
        metrics=metrics,
        confusion=_read_optional_csv(transfer / "confusion_matrix.csv"),
        classification_report=_read_optional_csv(transfer / "classification_report.csv"),
    )


def get_fsp_macro_columns(predictions: pd.DataFrame) -> tuple[List[str], List[str]]:
    """Retourne colonnes prob_* et dist_* présentes."""
    prob_cols = sorted([c for c in predictions.columns if c.startswith("prob_")])
    dist_cols = sorted([c for c in predictions.columns if c.startswith("dist_")])
    return prob_cols, dist_cols


def ensure_fsp_required_columns(predictions: pd.DataFrame, required: List[str]) -> None:
    """Valide la présence de colonnes attendues pour visualisation."""
    missing = [c for c in required if c not in predictions.columns]
    if missing:
        raise KeyError(f"Colonnes FSP manquantes: {missing}")


def plot_fsp_distribution_histograms(
    predictions: pd.DataFrame,
    *,
    fig_dir: Optional[Path] = None,
) -> None:
    """Histogrammes confidence/margin/entropy si colonnes disponibles."""
    fig_dir = Path(fig_dir) if fig_dir else None
    for col, title in (
        ("confidence", "Distribution de confidence"),
        ("margin", "Distribution de margin"),
        ("entropy", "Distribution de entropy"),
    ):
        if col not in predictions.columns:
            print(f"(absent) {col}")
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(
            pd.to_numeric(predictions[col], errors="coerce").dropna(),
            bins=40,
            color="steelblue",
            edgecolor="white",
        )
        ax.set_title(title)
        ax.set_xlabel(col)
        plt.tight_layout()
        if fig_dir:
            fig.savefig(fig_dir / f"hist_{col}.png", dpi=140, bbox_inches="tight")
        plt.show()
        plt.close(fig)


def plot_fsp_pred_macro_distribution(
    predictions: pd.DataFrame,
    *,
    fig_dir: Optional[Path] = None,
) -> None:
    """Barplot de la répartition des macros prédites."""
    if "pred_macro" not in predictions.columns:
        print("(absent) pred_macro")
        return
    counts = predictions["pred_macro"].astype(str).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color="#4c78a8")
    ax.set_title(STEP_DISTRIBUTION)
    ax.set_xlabel(STEP_PREDICTED)
    ax.set_ylabel("n")
    plt.tight_layout()
    if fig_dir:
        out = Path(fig_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "pred_macro_distribution.png", dpi=140, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_fsp_confusion_heatmap(
    confusion_df: pd.DataFrame,
    *,
    title: str = "Matrice de confusion",
    fig_dir: Optional[Path] = None,
) -> None:
    """Heatmap confusion_matrix.csv si disponible."""
    if confusion_df is None or confusion_df.empty:
        print("Matrice de confusion absente.")
        return
    cm = confusion_df.copy()
    if "true_macro" in cm.columns:
        cm = cm.set_index("true_macro")
    elif "Unnamed: 0" in cm.columns:
        cm = cm.set_index("Unnamed: 0")
    elif cm.columns.size > 0:
        # Cas fréquent: première colonne texte issue de l'index CSV.
        first_col = str(cm.columns[0])
        if not pd.api.types.is_numeric_dtype(cm[first_col]):
            cm = cm.set_index(first_col)
    # Garder uniquement des colonnes numériques pour seaborn.
    cm = cm.apply(pd.to_numeric, errors="coerce")
    cm = cm.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if cm.empty:
        print("Matrice de confusion vide après conversion numérique.")
        return
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Vrai")
    plt.tight_layout()
    if fig_dir:
        out = Path(fig_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "confusion_heatmap.png", dpi=140, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_fsp_distance_boxplot(
    predictions: pd.DataFrame,
    *,
    fig_dir: Optional[Path] = None,
) -> None:
    """Boxplot des distances par macro prédite."""
    if "pred_macro" not in predictions.columns:
        print("(absent) pred_macro")
        return
    _, dist_cols = get_fsp_macro_columns(predictions)
    if not dist_cols:
        print("(absent) colonnes dist_*")
        return
    long_df = predictions.melt(
        id_vars=["pred_macro"],
        value_vars=dist_cols,
        var_name="dist_macro",
        value_name="distance",
    )
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=long_df, x="pred_macro", y="distance", hue="dist_macro", ax=ax)
    ax.set_title(STEP_DISTANCE_BOX)
    ax.legend(title="distance")
    plt.tight_layout()
    if fig_dir:
        out = Path(fig_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "distance_boxplot_by_pred_macro.png", dpi=140, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def compute_fsp_confidence_calibration(
    predictions: pd.DataFrame,
    *,
    true_col: str = "true_macro",
    pred_col: str = "pred_macro",
    conf_col: str = "confidence",
    n_bins: int = 10,
) -> Optional[pd.DataFrame]:
    """Calibration simple: accuracy par quantile de confidence."""
    if true_col not in predictions.columns or pred_col not in predictions.columns or conf_col not in predictions.columns:
        return None
    df = predictions[[true_col, pred_col, conf_col]].copy()
    df = df.dropna()
    if df.empty:
        return None
    df["is_correct"] = (df[true_col].astype(str) == df[pred_col].astype(str)).astype(float)
    df["bin"] = pd.qcut(df[conf_col].astype(float), q=min(n_bins, len(df)), duplicates="drop")
    out = (
        df.groupby("bin", observed=True)
        .agg(
            n=("is_correct", "size"),
            mean_confidence=(conf_col, "mean"),
            accuracy=("is_correct", "mean"),
        )
        .reset_index()
    )
    return out


def get_fsp_top_confident_errors(
    predictions: pd.DataFrame,
    *,
    true_col: str = "true_macro",
    pred_col: str = "pred_macro",
    conf_col: str = "confidence",
    top_k: int = 20,
) -> pd.DataFrame:
    """Erreurs les plus confiantes (diagnostic)."""
    if true_col not in predictions.columns or pred_col not in predictions.columns:
        return pd.DataFrame()
    out = predictions.copy()
    out = out.loc[out[true_col].astype(str) != out[pred_col].astype(str)]
    if out.empty:
        return out
    if conf_col in out.columns:
        out = out.sort_values(conf_col, ascending=False)
    return out.head(top_k)


def _theme_label_map(themes: pd.DataFrame, *, max_chars: int = 52) -> Dict[Tuple[str, int], str]:
    """Libellés courts pour DataMapPlot (theme_label / theme_title / top_words)."""
    if themes.empty or "macro" not in themes.columns or "topic_id" not in themes.columns:
        return {}
    out: Dict[Tuple[str, int], str] = {}
    for _, row in themes.iterrows():
        macro = str(row["macro"])
        tid = int(row["topic_id"])
        raw = str(
            row.get("theme_label")
            or row.get("theme_title")
            or row.get("theme_summary")
            or row.get("top_words", "")
        ).strip()
        if len(raw) > max_chars:
            raw = raw[: max_chars - 1] + "…"
        out[(macro, tid)] = f"{macro}·T{tid}: {raw}" if raw else f"{macro}·T{tid}"
    return out


def _theme_label_plain_map(themes: pd.DataFrame) -> Dict[Tuple[str, int], str]:
    """Libellés courts pour légende / infobulle (sans préfixe macro·T)."""
    if themes.empty or "macro" not in themes.columns or "topic_id" not in themes.columns:
        return {}
    out: Dict[Tuple[str, int], str] = {}
    for _, row in themes.iterrows():
        macro = str(row["macro"])
        tid = int(row["topic_id"])
        raw = str(
            row.get("theme_label")
            or row.get("theme_title")
            or row.get("theme_summary")
            or row.get("top_words", "")
        ).strip()
        if raw and raw.lower() != "nan":
            out[(macro, tid)] = raw
    return out


def pick_accident_id_for_colored_text(
    df: pd.DataFrame,
    *,
    accident_id_col: str = "accident_id",
    min_units: int = 3,
    prefer_id: Any = None,
) -> Any:
    """Choisit un accident_id avec assez d'unités (évite les récits à 1–2 phrases)."""
    if accident_id_col not in df.columns:
        raise KeyError(f"Colonne {accident_id_col!r} absente")
    counts = df.groupby(accident_id_col, dropna=False).size().sort_values(ascending=False)
    if prefer_id is not None and prefer_id in set(counts.index):
        return prefer_id
    eligible = counts[counts >= min_units]
    if len(eligible):
        return eligible.index[0]
    if len(counts):
        return counts.index[0]
    raise ValueError("Aucun accident_id dans le DataFrame")


def merge_assignments(
    meta: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    confidence_threshold: float = 0.5,
    filter_meta_by_confidence: bool = True,
    macro_col: str = "m_hat",
) -> pd.DataFrame:
    """Joint meta + gating + topic_id (filtrage q_conf optionnel sur les lignes meta)."""
    df = meta.copy()
    if macro_col not in df.columns and "m_hat" in df.columns:
        macro_col = "m_hat"
    if filter_meta_by_confidence and "q_conf" in df.columns:
        df = df.loc[df["q_conf"].astype(float) >= confidence_threshold].copy()
    if assignments.empty or "doc_idx" not in assignments.columns:
        df["topic_id"] = -1
        return df.reset_index(drop=True)
    assign_cols = ["doc_idx", "macro", "topic_id"]
    for c in ("prob", "p_mk"):
        if c in assignments.columns:
            assign_cols.append(c)
    sub = assignments.loc[assignments["topic_id"] >= 0, assign_cols].copy()
    merged = df.copy()
    if "doc_idx" not in merged.columns:
        merged = merged.reset_index(drop=False).rename(columns={"index": "doc_idx"})
        if "doc_idx" not in merged.columns:
            merged["doc_idx"] = np.arange(len(merged))
    merged = merged.merge(sub, on="doc_idx", how="left")
    merged["topic_id"] = merged["topic_id"].fillna(-1).astype(int)
    return merged


def _resolve_macro_column(df: pd.DataFrame) -> str:
    for col in ("macro", "m_hat", "macro_y", "pred_label"):
        if col in df.columns:
            return col
    return "m_hat"


def build_topics_display_dataframe(
    out_dir: str | Path,
    meta: pd.DataFrame,
    *,
    confidence_threshold: float = 0.0,
    filter_meta_by_confidence: bool = False,
    topic_subdir: str = "topics_bertopic",
) -> pd.DataFrame:
    """
    Tableau unitaire prêt pour l'affichage texte coloré (toutes les phrases du corpus test).

    Joint meta FSP + ``assignments.csv`` (left join). Ajoute ``theme_label`` (long),
    ``theme_label_short`` (légende) et ``Probability`` (``prob`` ou ``p_mk``).
    """
    root = Path(out_dir).resolve() / topic_subdir
    themes_path = root / "themes_by_macro.csv"
    assign_path = root / "assignments.csv"
    if not assign_path.is_file():
        raise FileNotFoundError(f"Assignations BERTopic manquantes : {assign_path}")

    assignments = pd.read_csv(assign_path)
    merged = merge_assignments(
        meta,
        assignments,
        confidence_threshold=confidence_threshold,
        filter_meta_by_confidence=filter_meta_by_confidence,
    )
    themes = pd.read_csv(themes_path) if themes_path.is_file() else pd.DataFrame()
    tmap_long = _theme_label_map(themes, max_chars=240)
    tmap_short = _theme_label_plain_map(themes)

    macro_col = _resolve_macro_column(merged)
    if "macro" in merged.columns:
        merged["_macro_for_color"] = (
            merged["macro"].fillna(merged[macro_col]).astype(str).str.strip()
        )
    else:
        merged["_macro_for_color"] = merged[macro_col].astype(str).str.strip()

    def _label(row: pd.Series, short: bool) -> str:
        tid = int(row.get("topic_id", -1))
        if tid < 0:
            return ""
        key = (str(row["_macro_for_color"]), tid)
        if short:
            return tmap_short.get(key, tmap_long.get(key, f"{key[0]}·T{tid}"))
        return tmap_long.get(key, f"{key[0]}·T{tid}")

    merged["theme_label"] = merged.apply(lambda r: _label(r, short=False), axis=1)
    merged["theme_label_short"] = merged.apply(lambda r: _label(r, short=True), axis=1)

    if "prob" in merged.columns:
        merged["Probability"] = pd.to_numeric(merged["prob"], errors="coerce")
    elif "p_mk" in merged.columns:
        merged["Probability"] = pd.to_numeric(merged["p_mk"], errors="coerce")
    else:
        merged["Probability"] = np.nan

    return merged


def _rgba_to_hex(rgba: Tuple[float, ...]) -> str:
    r, g, b, _a = rgba
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def _hex_to_rgb(hexcol: str) -> Tuple[int, int, int]:
    hexcol = hexcol.lstrip("#")
    return tuple(int(hexcol[i : i + 2], 16) for i in (0, 2, 4))


def _blend_with_white(hexcol: str, strength: float = 0.82) -> str:
    r, g, b = _hex_to_rgb(hexcol)
    r2 = int(r + (255 - r) * strength)
    g2 = int(g + (255 - g) * strength)
    b2 = int(b + (255 - b) * strength)
    return f"rgb({r2},{g2},{b2})"


def _sort_accident_units(d: pd.DataFrame) -> pd.DataFrame:
    sort_cols: List[str] = []
    if "fact_id" in d.columns:
        sort_cols.append("fact_id")
    if "doc_idx" in d.columns:
        sort_cols.append("doc_idx")
    if sort_cols:
        return d.sort_values(sort_cols, kind="mergesort")
    return d.sort_index()


def _topic_style_map_for_accident(
    d: pd.DataFrame,
    macro_col: str,
    bertopic_id_col: str = "topic_id",
) -> Tuple[Dict[Tuple[str, int], int], List[Tuple[str, int]]]:
    pairs = sorted(
        {
            (str(row[macro_col]), int(row[bertopic_id_col]))
            for _, row in d.iterrows()
            if int(row.get(bertopic_id_col, -1)) >= 0
            and str(row.get(macro_col, "")).strip()
        }
    )
    pair_to_style = {p: i for i, p in enumerate(pairs)}
    return pair_to_style, pairs


def render_colored_accident_html(
    df: pd.DataFrame,
    accident_id: Any,
    *,
    accident_id_col: str = "accident_id",
    text_col: str = "sentence",
    bertopic_id_col: str = "topic_id",
    macro_col: str = "_macro_for_color",
    topic_label_col: str = "theme_label_short",
    prob_col: str = "Probability",
    min_prob: float = 0.0,
    drop_below_min_prob: bool = False,
    show_prob: bool = False,
    keep_outliers_plain: bool = True,
    add_tooltip: bool = True,
    show_legend: bool = True,
    legend_max_items: int = 80,
    cmap_name: str = "tab20",
    color_by: str = "topic",
    sep: str = " ",
    font_size_px: int = 11,
    line_height: float = 1.55,
    highlight_style: str = "border",
    pastel_strength: float = 0.82,
    border_width_px: int = 2,
    container_max_width_px: int = 1100,
    legend_font_size_px: int = 9,
    legend_title: str = "Thèmes",
    legend_max_height_px: int = 260,
    legend_item_min_width_px: int = 220,
) -> str:
    """
    HTML inline : toutes les unités textuelles d'un accident, surlignées par topic ou macro.

    ``color_by='topic'`` : une couleur par couple (macro, topic_id) présent dans le récit.
    """
    import html as html_lib

    if accident_id_col not in df.columns:
        raise KeyError(f"Colonne {accident_id_col!r} absente du DataFrame")

    d = _sort_accident_units(df.loc[df[accident_id_col] == accident_id].copy())
    if d.empty:
        return f"<p>Aucune donnée pour {accident_id_col}={accident_id}</p>"

    color_by_norm = str(color_by).lower().strip()
    if color_by_norm not in ("topic", "macro"):
        raise ValueError("color_by doit être 'topic' ou 'macro'")
    if legend_title == "Thèmes" and color_by_norm == "macro":
        legend_title = CHAIN_LEGEND

    pair_to_style: Dict[Tuple[str, int], int] = {}
    ordered_pairs: List[Tuple[str, int]] = []
    if color_by_norm == "topic":
        pair_to_style, ordered_pairs = _topic_style_map_for_accident(
            d, macro_col, bertopic_id_col=bertopic_id_col
        )

    n_colors = max(1, len(ordered_pairs) if color_by_norm == "topic" else len(MACRO_NAMES))
    cmap = plt.get_cmap(cmap_name, n_colors)
    color_hex_topic = {i: _rgba_to_hex(cmap(i)) for i in range(len(ordered_pairs))}
    pastel_bg_topic = {
        i: _blend_with_white(color_hex_topic[i], pastel_strength) for i in color_hex_topic
    }
    color_hex_macro = dict(MACRO_COLOR_HEX)
    pastel_bg_macro = {
        m: _blend_with_white(color_hex_macro.get(m, "#999999"), pastel_strength)
        for m in MACRO_NAMES
    }

    def get_topic_label(row: pd.Series, macro: str, tid: int) -> str:
        for col in (topic_label_col, "theme_label_short", "theme_label"):
            if col and col in d.columns:
                val = row.get(col, "")
                if val is not None:
                    s = str(val).strip()
                    if s and s.lower() != "nan":
                        return s
        return f"{macro}·T{tid}" if tid >= 0 else macro

    parts: List[str] = []
    for _, row in d.iterrows():
        sent = row.get(text_col, "")
        sent = "" if sent is None else str(sent).strip()
        if not sent:
            continue

        macro = str(row.get(macro_col, row.get("m_hat", ""))).strip()
        tid = int(row.get(bertopic_id_col, -1)) if row.get(bertopic_id_col) is not None else -1
        label = get_topic_label(row, macro, tid)

        p_val: Optional[float] = None
        if prob_col in d.columns:
            try:
                p = row.get(prob_col)
                p_val = float(p) if p is not None and str(p) != "nan" else None
            except (TypeError, ValueError):
                p_val = None

        if min_prob is not None and p_val is not None and p_val < float(min_prob):
            if drop_below_min_prob:
                continue
            style_key: Any = -1
        elif color_by_norm == "macro":
            style_key = macro if macro in MACRO_COLOR_HEX else "other"
        else:
            style_key = pair_to_style.get((macro, tid), -1) if tid >= 0 else -1

        sent_esc = html_lib.escape(sent)
        prob_html = ""
        if show_prob and p_val is not None:
            prob_html = (
                f' <span style="color:#666;font-size:0.95em;">(p={p_val:.2f})</span>'
            )

        use_plain = (color_by_norm == "topic" and style_key == -1) or (
            color_by_norm == "macro" and style_key == "other"
        )
        if use_plain and keep_outliers_plain:
            parts.append(f'<span style="color:#111;">{sent_esc}{prob_html}</span>')
            continue

        if color_by_norm == "macro":
            col = color_hex_macro.get(str(style_key), "#999999")
            bg = pastel_bg_macro.get(str(style_key), "#f5f5f5")
        else:
            col = color_hex_topic.get(int(style_key), "#999999")
            bg = pastel_bg_topic.get(int(style_key), "#f5f5f5")

        title = html_lib.escape(label) if add_tooltip else ""
        if highlight_style == "bar":
            style = (
                f"background:transparent;border-left:{border_width_px}px solid {col};"
                f"padding:1px 7px;border-radius:8px;"
            )
        elif highlight_style == "bg":
            style = (
                f"background:{bg};padding:1px 6px;border-radius:8px;"
                f"border:1px solid rgba(0,0,0,0.06);"
            )
        else:
            style = (
                f"background:{bg};padding:1px 6px;border-radius:8px;"
                f"border:1px solid rgba(0,0,0,0.06);"
                f"box-shadow:0 0 0 {border_width_px}px {col} inset;"
            )
        parts.append(f"<span title='{title}' style='{style}'>{sent_esc}{prob_html}</span>")

    html_text = sep.join(parts)

    legend_html = ""
    if show_legend:
        legend_items: List[Tuple[Any, str, str, str]] = []
        if color_by_norm == "macro":
            for m in MACRO_NAMES:
                if (d[macro_col].astype(str).str.strip() == m).any():
                    legend_items.append(
                        (m, MACRO_COLOR_HEX.get(m, "#999"), pastel_bg_macro.get(m, "#f5f5f5"), m)
                    )
        else:
            for pair in ordered_pairs[:legend_max_items]:
                sk = pair_to_style[pair]
                row_ex = d.loc[
                    (d[macro_col].astype(str).str.strip() == pair[0])
                    & (d[bertopic_id_col].astype(int) == pair[1])
                ].head(1)
                lab = (
                    get_topic_label(row_ex.iloc[0], pair[0], pair[1])
                    if not row_ex.empty
                    else f"{pair[0]}·T{pair[1]}"
                )
                legend_items.append(
                    (
                        sk,
                        color_hex_topic.get(sk, "#999"),
                        pastel_bg_topic.get(sk, "#f5f5f5"),
                        lab,
                    )
                )

        if legend_items:
            items_html = []
            for _key, col, bg, lab_full in legend_items:
                lab_esc = html_lib.escape(str(lab_full))
                tip = f' title="{lab_esc}"' if add_tooltip else ""
                items_html.append(
                    f'<div{tip} style="display:flex;align-items:flex-start;gap:8px;'
                    f"padding:5px 7px;border-radius:10px;background:{bg};"
                    f"border:1px solid rgba(0,0,0,0.06);"
                    f'min-width:{legend_item_min_width_px}px;">'
                    f'<div style="width:10px;height:10px;background:{col};border-radius:3px;'
                    f'margin-top:2px;flex:0 0 auto;"></div>'
                    f'<div style="font-size:{legend_font_size_px}px;color:#1a1a1a;'
                    f"font-weight:500;line-height:1.25;white-space:normal;"
                    f'overflow-wrap:anywhere;word-break:break-word;">{lab_esc}</div></div>'
                )
            more = ""
            if color_by_norm == "topic" and len(ordered_pairs) > legend_max_items:
                more = (
                    f'<div style="color:#666;font-size:{legend_font_size_px}px;margin-top:6px;">'
                    f"… +{len(ordered_pairs) - legend_max_items} thèmes</div>"
                )
            legend_html = (
                f'<div style="margin-top:10px;padding:10px 12px;border:1px solid #eee;'
                f'border-radius:14px;background:#fbfbfb;">'
                f'<div style="font-weight:800;margin-bottom:8px;color:#111;">'
                f'{html_lib.escape(legend_title)}</div>'
                f'<div style="display:flex;flex-wrap:wrap;gap:8px;'
                f'max-height:{legend_max_height_px}px;overflow:auto;padding-right:6px;">'
                f'{"".join(items_html)}{more}</div></div>'
            )

    return (
        f'<div style="max-width:{container_max_width_px}px;">'
        f'<div style="font-size:{font_size_px}px;line-height:{line_height};'
        f'white-space:pre-wrap;color:#111;border:1px solid #e0e0e0;'
        f'padding:12px;border-radius:14px;background:#fff;">{html_text}</div>'
        f'{legend_html}</div>'
    )


def show_colored_text_inline(
    df: pd.DataFrame,
    accident_id: Any,
    **kwargs: Any,
) -> Optional[str]:
    """Affiche le récit coloré dans Jupyter ; retourne le HTML généré."""
    html = render_colored_accident_html(df, accident_id, **kwargs)
    try:
        from IPython.display import HTML, display

        display(HTML(html))
    except ImportError:
        print(html[:2000])
    return html


def show_colored_text_for_accident(
    out_dir: str | Path,
    meta: pd.DataFrame,
    accident_id: Any,
    *,
    confidence_threshold: float = 0.0,
    **kwargs: Any,
) -> None:
    """Charge meta+topics depuis un run FSP et affiche le récit coloré."""
    df = build_topics_display_dataframe(
        out_dir,
        meta,
        confidence_threshold=confidence_threshold,
    )
    show_colored_text_inline(df, accident_id, **kwargs)


_TEST_LABEL_COL_CANDIDATES = ("pred_label", "true_macro", "pred_macro", "m_hat", "label")


@dataclass
class RawTestEmbeddingVizResult:
    """Résultat de plot_test_corpus_raw_embeddings."""

    pca_tsne_path: Optional[Path] = None
    tsne_per_macro_path: Optional[Path] = None
    umap_png_path: Optional[Path] = None
    umap_html_path: Optional[Path] = None
    metrics_path: Optional[Path] = None
    metrics_df: Optional[pd.DataFrame] = None
    label_col: Optional[str] = None
    n_points: int = 0
    missing: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.missing is None:
            self.missing = []


def resolve_test_label_col(meta: pd.DataFrame, label_col: Optional[str] = None) -> str:
    """Retourne la colonne étape (chaîne accidentelle) pour colorer les cartes."""
    if label_col and label_col in meta.columns:
        return label_col
    for col in _TEST_LABEL_COL_CANDIDATES:
        if col in meta.columns:
            return col
    raise KeyError(
        "Aucune colonne étape trouvée "
        f"(essayé : {', '.join(_TEST_LABEL_COL_CANDIDATES)} ; "
        f"colonnes : {list(meta.columns)[:12]}…)"
    )


def _make_matplotlib_save_fig(fig_dir: Path, *, show: bool, dpi: int = 120):
    """Fabrique un callback save_fig compatible scgm_text.notebook_viz."""
    out = Path(fig_dir)
    out.mkdir(parents=True, exist_ok=True)

    def save_fig(name: str) -> Path:
        path = out / name
        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()
        return path

    return save_fig


def plot_test_corpus_raw_embeddings(
    test_corpus: str,
    *,
    fig_dir: Path,
    anchor: Optional[Path] = None,
    label_col: Optional[str] = None,
    max_points: int = 12000,
    seed: int = 42,
    prefix: str = "raw_test_embedding",
    show: bool = True,
    include_umap: bool = True,
    display_metrics: bool = True,
) -> RawTestEmbeddingVizResult:
    """
    PCA + t-SNE (+ UMAP) sur les embeddings encodeur Qwen du corpus test.

    Utilise ``configs/test_corpora.yaml`` (``data_csv`` + ``emb_csv``).
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    from safer_core.paths import TEXT_ROOT
    from safer_core.test_corpus import raw_embedding_test_dir, resolve_test_corpus
    from scgm_text.dataset_text_embeddings import merge_metadata_with_embeddings
    from scgm_text.notebook_viz import (
        plot_embedding_umap_by_macro,
        plot_projection_matplotlib,
        plot_tsne_per_macro_grid,
        sample_projection_indices,
    )
    from scgm_text.utils_io import create_doc_id_if_missing

    root = Path(anchor or TEXT_ROOT).resolve()
    try:
        spec = resolve_test_corpus(test_corpus, anchor=root)
    except KeyError as exc:
        result = RawTestEmbeddingVizResult()
        result.missing.append(str(exc))
        return result
    result = RawTestEmbeddingVizResult()
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    data_path = spec.data_csv
    emb_path = spec.emb_csv
    if not data_path.is_file():
        result.missing.append(str(data_path))
        return result
    if not emb_path.is_file():
        result.missing.append(str(emb_path))
        return result

    meta = create_doc_id_if_missing(pd.read_csv(data_path))
    try:
        resolved_label = resolve_test_label_col(meta, label_col)
    except KeyError as exc:
        result.missing.append(str(exc))
        return result

    slim = meta.drop(columns=[c for c in meta.columns if c.startswith("dim_")], errors="ignore")
    try:
        merged, dim_cols = merge_metadata_with_embeddings(slim, str(emb_path))
    except Exception as exc:
        result.missing.append(f"merge embeddings : {exc}")
        return result

    raw = merged[dim_cols].to_numpy(dtype=np.float64)
    idx = sample_projection_indices(meta, resolved_label, max_points=max_points, seed=seed)
    sample_df = meta.loc[idx]
    sample_x = raw[idx]

    pca_xy = PCA(n_components=2, random_state=seed).fit_transform(sample_x)
    tsne_xy = TSNE(
        n_components=2,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    ).fit_transform(sample_x)

    save_fig = _make_matplotlib_save_fig(fig_dir, show=show)
    corpus_label = spec.display_name
    result.label_col = resolved_label
    result.n_points = len(sample_df)
    result.pca_tsne_path = plot_projection_matplotlib(
        pca_xy,
        tsne_xy,
        sample_df,
        resolved_label,
        save_fig=save_fig,
        png_name=f"{prefix}_pca_tsne.png",
        pca_title=f"PCA 2D — {corpus_label} (embedding brut)",
        tsne_title=f"t-SNE 2D — {corpus_label} (embedding brut)",
        show_macro_centroids=True,
        show_z_centroids=False,
    )
    result.tsne_per_macro_path = plot_tsne_per_macro_grid(
        sample_x,
        sample_df[resolved_label].astype(str).to_numpy(),
        corpus_name=f"{corpus_label} (embedding brut)",
        save_fig=save_fig,
        png_name=f"{prefix}_tsne_per_macro.png",
        seed=seed,
        max_points_per_macro=min(2000, max(10, max_points // 4)),
    )

    if include_umap:
        umap_paths = plot_embedding_umap_by_macro(
            raw,
            meta,
            resolved_label,
            figures_dir=fig_dir,
            save_fig=save_fig,
            max_points=max_points,
            seed=seed,
            title=f"UMAP — {corpus_label} (embedding brut, couleur = étape)",
            png_name=f"{prefix}_umap.png",
            html_name=f"{prefix}_umap_interactive.html",
        )
        for p in umap_paths:
            if p.suffix.lower() == ".png":
                result.umap_png_path = p
            elif p.suffix.lower() == ".html":
                result.umap_html_path = p

    metrics_path = raw_embedding_test_dir(test_corpus, anchor=root) / "metrics" / "metrics_geometry.csv"
    if metrics_path.is_file():
        result.metrics_path = metrics_path
        result.metrics_df = pd.read_csv(metrics_path)
        if display_metrics:
            try:
                from IPython.display import display

                print(f"=== Géométrie embedding brut — {corpus_label} ===")
                display(result.metrics_df)
            except ImportError:
                print(result.metrics_df.to_string(index=False))

    return result


RAW_TEST_EMBEDDING_SECTION_MD = (
    "## Embedding brut — corpus test (PCA / t-SNE / UMAP)\n\n"
    "Vecteurs encodeur Qwen du registre test (`configs/test_corpora.yaml`), "
    "couleur = **étape** de la chaîne accidentelle. "
    "Métriques géométrie : `output_test/<corpus>/raw_embedding/metrics/metrics_geometry.csv` "
    "(job `export_raw_geometry.sh`)."
)


def notebook_raw_test_embedding_source(
    fig_dir: str = "FIG_DIR",
    *,
    display_metrics: bool = True,
) -> str:
    """Code source pour une cellule notebook (réutilisé par les builders)."""
    metrics_flag = "True" if display_metrics else "False"
    return f"""
from macro_transfer.notebook_viz import plot_test_corpus_raw_embeddings

_raw_emb = plot_test_corpus_raw_embeddings(
    TEST_CORPUS,
    fig_dir={fig_dir},
    anchor=TEXT_ROOT,
    max_points=12000,
    seed=42,
    prefix="raw_test_embedding",
    show=True,
    display_metrics={metrics_flag},
)
if _raw_emb.missing:
    print("Embedding brut test — fichiers manquants :", ", ".join(_raw_emb.missing))
else:
    print(
        "Figures embedding brut :",
        _raw_emb.pca_tsne_path,
        _raw_emb.tsne_per_macro_path,
        _raw_emb.umap_png_path,
    )
""".strip()
