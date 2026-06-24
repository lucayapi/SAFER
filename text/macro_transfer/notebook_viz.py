"""Visualisations notebook pour macro_transfer (UMAP, DataMapPlot, PCA/t-SNE)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    from macro_transfer.fsp_config import FSP_LEGACY_OUTPUT_DIR_ALIASES

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
        if not transfer.is_dir() and root.parent.name == "frozen_source_prototypes":
            legacy_suffix = FSP_LEGACY_OUTPUT_DIR_ALIASES.get(root.name)
            if legacy_suffix:
                cand_root = root.parent / legacy_suffix
                cand_transfer = cand_root / "transfer"
                if cand_transfer.is_dir():
                    root = cand_root
                    transfer = cand_transfer
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


def load_supervised_macro_ft_run_artifacts(out_dir: str | Path) -> FSPRunArtifacts:
    """Charge les sorties macro_transfer/supervised_macro_ft (sans prototypes FSP)."""
    root = Path(out_dir).resolve()
    transfer = root / "transfer"
    preds_path = transfer / "target_macro_predictions.csv"
    if not preds_path.is_file():
        raise FileNotFoundError(f"Prédictions manquantes : {preds_path}")
    metrics: Dict[str, Any] = {}
    metrics_path = transfer / "metrics.json"
    if metrics_path.is_file():
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
    protos = pd.DataFrame()
    return FSPRunArtifacts(
        out_dir=root,
        transfer_dir=transfer,
        predictions=pd.read_csv(preds_path),
        prototypes=protos,
        metrics=metrics,
        confusion=_read_optional_csv(transfer / "confusion_matrix.csv"),
        classification_report=_read_optional_csv(transfer / "classification_report.csv"),
    )


def load_supervised_macro_ft_vs_baseline07_metrics(corpus_id: str, *, anchor: Path) -> pd.DataFrame:
    """Compare métriques FT neural vs baseline 07 sklearn."""
    from supervised_macro_ft.transfer import supervised_macro_ft_output_dir

    rows: list[dict[str, Any]] = []

    ft_root = supervised_macro_ft_output_dir(corpus_id, anchor=anchor)
    ft_metrics = _load_fsp_metrics_json(ft_root)
    rows.append(
        {
            "Méthode": "Supervised macro FT (CE)",
            "Bal. Acc.": _fsp_metric_to_float(ft_metrics.get("balanced_accuracy")),
            "F1 (étapes)": _fsp_metric_to_float(ft_metrics.get("macro_f1")),
            "Confiance moy.": _fsp_metric_to_float(ft_metrics.get("mean_confidence")),
            "Entropie moy.": _fsp_metric_to_float(ft_metrics.get("mean_entropy")),
            "metrics_available": bool(ft_metrics),
        }
    )

    base_root = anchor / "output_test" / corpus_id / "supervised_baseline"
    base_metrics: dict[str, Any] = {}
    manifest_path = base_root / "run_manifest.json"
    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            best = json.load(f).get("best_model")
        best_path = base_root / "transfer" / "models" / str(best) / "metrics.json"
        if best_path.is_file():
            with open(best_path, encoding="utf-8") as f:
                base_metrics = json.load(f)
    rows.append(
        {
            "Méthode": "Baseline 07 (sklearn Qwen brut)",
            "Bal. Acc.": _fsp_metric_to_float(base_metrics.get("balanced_accuracy")),
            "F1 (étapes)": _fsp_metric_to_float(base_metrics.get("macro_f1")),
            "Confiance moy.": _fsp_metric_to_float(base_metrics.get("mean_confidence")),
            "Entropie moy.": _fsp_metric_to_float(base_metrics.get("mean_entropy")),
            "metrics_available": bool(base_metrics),
        }
    )
    return pd.DataFrame(rows)


def _fsp_metric_to_float(value: Any) -> float:
    try:
        x = float(value)
        if np.isnan(x):
            return np.nan
        return x
    except (TypeError, ValueError):
        return np.nan


def load_fsp_metrics_comparison_table(
    corpus_id: str,
    *,
    methods: Optional[Sequence[str]] = None,
    anchor: Path,
) -> pd.DataFrame:
    """Table comparative des métriques transfer/metrics.json pour chaque encodeur FSP."""
    from macro_transfer.fsp_config import (
        FSP_ENCODER_METHODS,
        resolve_fsp_method_display_name,
        resolve_fsp_output_dir,
    )

    use_methods = list(methods or FSP_ENCODER_METHODS)
    rows: List[Dict[str, Any]] = []
    for method in use_methods:
        out_dir = resolve_fsp_output_dir(corpus_id, method, anchor=anchor)
        metrics_path = out_dir / "transfer" / "metrics.json"
        metrics: Dict[str, Any] = {}
        if metrics_path.is_file():
            with open(metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)
        label = metrics.get("method") or resolve_fsp_method_display_name(method)
        rows.append(
            {
                "method_key": method,
                "Méthode": str(label),
                "Bal. Acc.": _fsp_metric_to_float(metrics.get("balanced_accuracy", np.nan)),
                F1_STEPS: _fsp_metric_to_float(metrics.get("macro_f1", np.nan)),
                "Confiance moy.": _fsp_metric_to_float(metrics.get("mean_confidence", np.nan)),
                "Entropie moy.": _fsp_metric_to_float(metrics.get("mean_entropy", np.nan)),
                "metrics_available": metrics_path.is_file(),
                "out_dir": str(out_dir),
            }
        )
    return pd.DataFrame(rows)


def export_fsp_metrics_latex_table(table_df: pd.DataFrame) -> str:
    """Exporte un tableau LaTeX avec meilleures valeurs en gras."""
    metric_cols = ["Bal. Acc.", F1_STEPS, "Confiance moy.", "Entropie moy."]
    display_df = table_df[["Méthode", *metric_cols]].copy()

    def _winner_indices(values: Sequence[Any], *, mode: str = "max") -> set[int]:
        s = pd.Series(values, dtype="float64")
        if s.notna().sum() == 0:
            return set()
        best = s.max() if mode == "max" else s.min()
        return set(s.index[s == best].tolist())

    def _fmt(value: Any, bold: bool = False) -> str:
        if pd.isna(value):
            return "--"
        txt = f"{float(value):.4f}"
        return f"\\textbf{{{txt}}}" if bold else txt

    best_bal = _winner_indices(display_df["Bal. Acc."], mode="max")
    best_f1 = _winner_indices(display_df[F1_STEPS], mode="max")
    best_conf = _winner_indices(display_df["Confiance moy."], mode="max")
    best_ent = _winner_indices(display_df["Entropie moy."], mode="min")

    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Méthode} & \\textbf{Bal. Acc.} & "
        f"\\textbf{{{F1_STEPS}}} & \\textbf{{Confiance moy.}} & \\textbf{{Entropie moy.}} \\\\",
        "\\midrule",
    ]
    for i, row in display_df.iterrows():
        lines.append(
            f"{row['Méthode']} & "
            f"{_fmt(row['Bal. Acc.'], i in best_bal)} & "
            f"{_fmt(row[F1_STEPS], i in best_f1)} & "
            f"{_fmt(row['Confiance moy.'], i in best_conf)} & "
            f"{_fmt(row['Entropie moy.'], i in best_ent)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def plot_fsp_methods_metrics_comparison(
    table_df: pd.DataFrame,
    *,
    fig_dir: Optional[Path] = None,
    show: bool = True,
) -> Optional[Path]:
    """Barres Bal. Acc. et F1 pour toutes les méthodes FSP disponibles."""
    if "metrics_available" in table_df.columns:
        plot_df = table_df[table_df["metrics_available"]].copy()
    else:
        plot_df = table_df.copy()
    if plot_df.empty:
        return None

    labels = plot_df["Méthode"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(labels) * 1.2), 4))
    for ax, col, title in (
        (axes[0], "Bal. Acc.", "Balanced accuracy"),
        (axes[1], F1_STEPS, F1_STEPS),
    ):
        vals = plot_df[col].astype(float)
        ax.bar(x, vals, color="#4c78a8")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(title)

    fig.tight_layout()
    out_path: Optional[Path] = None
    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_path = fig_dir / "fsp_methods_metrics_comparison.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


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
    filename: str = "confusion_heatmap.png",
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
        fig.savefig(out / filename, dpi=140, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_tsne_true_vs_pred(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    *,
    title: str = "",
    fig_dir: Optional[Path] = None,
    filename: str = "tsne_true_vs_pred.png",
    max_points: int = 2000,
    seed: int = 42,
    macros: Sequence[str] = MACRO_NAMES,
) -> Optional[Path]:
    """
    t-SNE 2D : panneau gauche = vraie classe, droite = classe prédite (mêmes coordonnées).
    """
    from sklearn.manifold import TSNE

    h = np.asarray(embeddings, dtype=np.float64)
    if len(h) != len(df):
        raise ValueError(f"embeddings ({len(h)}) et metadata ({len(df)}) non alignés")
    if true_col not in df.columns or pred_col not in df.columns:
        raise KeyError(f"Colonnes manquantes : {true_col!r}, {pred_col!r}")

    n = len(h)
    if n > int(max_points):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(n, size=int(max_points), replace=False)
    else:
        idx = np.arange(n, dtype=np.int64)
    X = h[idx]
    sub = df.iloc[idx].reset_index(drop=True)

    tsne_xy = TSNE(
        n_components=2,
        random_state=int(seed),
        init="pca",
        learning_rate="auto",
    ).fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, subtitle in zip(
        axes,
        (true_col, pred_col),
        ("Vraie classe", "Classe prédite"),
    ):
        for macro in macros:
            mask = sub[col].astype(str).values == str(macro)
            if not mask.any():
                continue
            ax.scatter(
                tsne_xy[mask, 0],
                tsne_xy[mask, 1],
                s=12,
                alpha=0.55,
                label=str(macro),
                c=MACRO_COLOR_HEX.get(str(macro), "gray"),
            )
        ax.set_title(f"{subtitle}" + (f" — {title}" if title else ""))
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(markerscale=2, fontsize=8, loc="best")
    plt.tight_layout()
    out_path = None
    if fig_dir:
        out = Path(fig_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / filename
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return out_path


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


def _load_fsp_metrics_json(out_dir: Path) -> dict[str, Any]:
    metrics_path = out_dir / "transfer" / "metrics.json"
    if not metrics_path.is_file():
        return {}
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


def load_softtriple_native_vs_legacy_metrics(
    corpus_id: str,
    *,
    anchor: Path,
) -> pd.DataFrame:
    """Compare métriques FSP softtriple_native vs softtriple (prototype moyen)."""
    from macro_transfer.fsp_config import (
        FSP_SOFTTRIPLE_NATIVE_METHOD,
        resolve_fsp_output_dir,
    )

    rows: list[dict[str, Any]] = []
    for method_key, label in (
        (FSP_SOFTTRIPLE_NATIVE_METHOD, "SoftTriple (centres natifs)"),
        ("softtriple", "SoftTriple (prototype moyen)"),
    ):
        out_dir = resolve_fsp_output_dir(corpus_id, method_key, anchor=anchor)
        metrics = _load_fsp_metrics_json(out_dir)
        rows.append(
            {
                "method_key": method_key,
                "Méthode": label,
                "Bal. Acc.": _fsp_metric_to_float(metrics.get("balanced_accuracy")),
                "F1 (étapes)": _fsp_metric_to_float(metrics.get("macro_f1")),
                "Confiance moy.": _fsp_metric_to_float(metrics.get("mean_confidence")),
                "Entropie moy.": _fsp_metric_to_float(metrics.get("mean_entropy")),
                "assignment_mode": metrics.get("assignment_mode"),
                "gamma": metrics.get("gamma"),
                "temperature": metrics.get("temperature"),
                "distance_metric": metrics.get("distance_metric"),
                "centers_per_class": metrics.get("centers_per_class"),
                "metrics_available": bool(metrics),
                "out_dir": str(out_dir),
            }
        )
    df = pd.DataFrame(rows)
    if len(df) == 2 and df["metrics_available"].all():
        native = df.iloc[0]
        legacy = df.iloc[1]
        df.loc[len(df)] = {
            "method_key": "delta_native_minus_legacy",
            "Méthode": "Δ (natif − legacy)",
            "Bal. Acc.": native["Bal. Acc."] - legacy["Bal. Acc."],
            "F1 (étapes)": native["F1 (étapes)"] - legacy["F1 (étapes)"],
            "Confiance moy.": native["Confiance moy."] - legacy["Confiance moy."],
            "Entropie moy.": native["Entropie moy."] - legacy["Entropie moy."],
            "metrics_available": True,
        }
    return df


def _prototype_dim_matrix(prototypes: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    dim_cols = [c for c in prototypes.columns if c.startswith("dim_")]
    if not dim_cols:
        raise ValueError("Colonnes dim_* absentes dans source_prototypes.csv")
    labels: list[str] = []
    if "macro" in prototypes.columns and "center_k" in prototypes.columns:
        for _, row in prototypes.iterrows():
            labels.append(f"{row['macro']}_k{int(row['center_k'])}")
    else:
        labels = [f"c{i}" for i in range(len(prototypes))]
    mat = prototypes[dim_cols].to_numpy(dtype=np.float64)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    mat = mat / norms
    return mat, labels


def plot_softtriple_center_similarity_heatmap(
    prototypes: pd.DataFrame,
    *,
    fig_dir: Optional[Path] = None,
    filename: str = "softtriple_center_similarity_heatmap.png",
) -> None:
    """Heatmap cosinus entre centres W_{r,k}."""
    import seaborn as sns

    mat, labels = _prototype_dim_matrix(prototypes)
    sim = mat @ mat.T
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.45), max(5, len(labels) * 0.4)))
    sns.heatmap(sim, xticklabels=labels, yticklabels=labels, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Similarité cosinus entre centres SoftTriple")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if fig_dir:
        out = Path(fig_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / filename, dpi=140, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_softtriple_center_weight_bars(
    summary_df: pd.DataFrame,
    *,
    fig_dir: Optional[Path] = None,
    filename: str = "softtriple_center_weight_bars.png",
) -> None:
    """Barplot des poids moyens alpha par macro et centre k."""
    import seaborn as sns

    if summary_df.empty or "mean_weight" not in summary_df.columns:
        print("(absent) softtriple_center_weights_summary")
        return
    df = summary_df.copy()
    if "center_k" in df.columns and "macro" in df.columns:
        df["label"] = df["macro"].astype(str) + "_k" + df["center_k"].astype(int).astype(str)
    else:
        df["label"] = df.index.astype(str)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="label", y="mean_weight", hue="macro" if "macro" in df.columns else None, ax=ax)
    ax.set_title("Poids moyens des centres (alpha) sur le corpus cible")
    ax.set_xlabel("centre")
    ax.set_ylabel("mean_weight")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if fig_dir:
        out = Path(fig_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / filename, dpi=140, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_softtriple_relaxed_score_boxplot(
    predictions: pd.DataFrame,
    *,
    fig_dir: Optional[Path] = None,
    filename: str = "softtriple_relaxed_score_boxplot.png",
) -> None:
    """Boxplot des scores agrégés S (via dist_* = -S) par macro prédite."""
    import seaborn as sns

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
        var_name="score_macro",
        value_name="neg_relaxed_score",
    )
    long_df["relaxed_score"] = -pd.to_numeric(long_df["neg_relaxed_score"], errors="coerce")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=long_df, x="pred_macro", y="relaxed_score", hue="score_macro", ax=ax)
    ax.set_title("Scores relaxés S par macro (dist_* = -S)")
    ax.legend(title="macro score")
    plt.tight_layout()
    if fig_dir:
        out = Path(fig_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / filename, dpi=140, bbox_inches="tight")
    plt.show()
    plt.close(fig)


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


def resolve_bertopic_datamap_path(
    out_dir: str | Path,
    macro: str,
) -> Optional[Path]:
    """Chemin PNG DataMapPlot BERTopic pour une macro (figures/ puis bertopic/)."""
    root = Path(out_dir).resolve()
    for candidate in (
        root / "figures" / f"bertopic_datamap_{macro}.png",
        root / "bertopic" / str(macro) / "datamap_topics.png",
    ):
        if candidate.is_file():
            return candidate
    return None


def list_bertopic_datamap_paths(
    out_dir: str | Path,
    macros: Optional[Sequence[str]] = None,
) -> Dict[str, Path]:
    """Carte macro → PNG DataMapPlot si le fichier existe."""
    macro_list = list(macros) if macros is not None else list(MACRO_NAMES)
    out: Dict[str, Path] = {}
    for macro in macro_list:
        path = resolve_bertopic_datamap_path(out_dir, str(macro))
        if path is not None:
            out[str(macro)] = path
    return out


def show_bertopic_datamaps_inline(
    out_dir: str | Path,
    macros: Optional[Sequence[str]] = None,
) -> Dict[str, Path]:
    """Affiche les DataMapPlot BERTopic par macro dans Jupyter."""
    paths = list_bertopic_datamap_paths(out_dir, macros=macros)
    try:
        from IPython.display import Image, display
    except ImportError:
        for macro, path in paths.items():
            print(f"{macro}: {path}")
        return paths
    for macro in (macros or MACRO_NAMES):
        macro_s = str(macro)
        path = paths.get(macro_s)
        if path is None:
            print(f"DataMapPlot absent pour macro {macro_s}")
            continue
        print(f"Macro {macro_s} — {path.name}")
        display(Image(filename=str(path)))
    return paths


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


TOPIC_JUDGE_SECTION_MD = (
    "## Évaluation LLM-judge des topics BERTopic\n\n"
    "Chaque topic `(macro, topic_id)` est noté par un LLM sur **6 critères** (0–5) : "
    "cohérence interne, homogénéité accidentologique, alignement au rôle de l'étape, "
    "spécificité, nommabilité, utilité pour la reconstruction de scénario. "
    "Le **`score_global`** est la moyenne Python des 6 scores (non calculée par le LLM). "
    "Verdict : `conserver`, `fusionner`, `scinder`, `rejeter`.\n\n"
    "Sorties : `summary/topic_judge_scores.csv`, `summary/topic_judge_macro_summary.csv`."
)


def load_topic_judge_artifacts(out_dir: str | Path) -> Dict[str, Any]:
    """Charge scores judge + agrégats macro (DataFrames vides si absents)."""
    from macro_transfer.topic_judge import (
        load_topic_judge_macro_summary,
        load_topic_judge_scores,
        topic_judge_macro_summary_path,
        topic_judge_scores_path,
    )

    root = Path(out_dir).resolve()
    scores = load_topic_judge_scores(root)
    macro_summary = load_topic_judge_macro_summary(root)
    return {
        "scores": scores,
        "macro_summary": macro_summary,
        "scores_path": topic_judge_scores_path(root),
        "macro_summary_path": topic_judge_macro_summary_path(root),
    }


def plot_topic_judge_quality(
    out_dir: str | Path,
    *,
    fig_dir: Optional[Path] = None,
    score_threshold: float = 3.0,
    show: bool = True,
) -> Dict[str, Optional[Path]]:
    """
    Boxplot ``score_global`` par macro + barres empilées des verdicts.

    Écrit ``topic_judge_score_by_macro.png`` et ``topic_judge_verdict_by_macro.png``.
    """
    artifacts = load_topic_judge_artifacts(out_dir)
    scores = artifacts["scores"]
    out_paths: Dict[str, Optional[Path]] = {
        "score_by_macro": None,
        "verdict_by_macro": None,
    }
    if scores.empty or "score_global" not in scores.columns:
        print("topic_judge_scores.csv absent ou vide.")
        return out_paths

    df = scores.copy()
    df["macro"] = df["macro"].astype(str)
    df["score_global"] = pd.to_numeric(df["score_global"], errors="coerce")
    df = df.dropna(subset=["score_global"])
    if df.empty:
        print("Aucun score_global exploitable.")
        return out_paths

    macro_order = [m for m in MACRO_NAMES if m in set(df["macro"])]
    if not macro_order:
        macro_order = sorted(df["macro"].unique())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [df.loc[df["macro"] == m, "score_global"].to_numpy() for m in macro_order]
    ax.boxplot(data, labels=macro_order, patch_artist=True)
    ax.axhline(score_threshold, color="#d62728", linestyle="--", linewidth=1.2, label=f"seuil {score_threshold}")
    ax.set_title("Score global judge LLM par étape")
    ax.set_xlabel("Étape")
    ax.set_ylabel("score_global (0–5)")
    ax.set_ylim(0, 5.2)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    if fig_dir:
        fig_dir_p = Path(fig_dir)
        fig_dir_p.mkdir(parents=True, exist_ok=True)
        score_png = fig_dir_p / "topic_judge_score_by_macro.png"
        fig.savefig(score_png, dpi=140, bbox_inches="tight")
        out_paths["score_by_macro"] = score_png
    if show:
        plt.show()
    plt.close(fig)

    verdict_df = df.copy()
    verdict_df["verdict"] = verdict_df["verdict"].astype(str).str.lower()
    verdict_order = ["conserver", "fusionner", "scinder", "rejeter"]
    verdict_colors = {
        "conserver": "#2ca02c",
        "fusionner": "#ff7f0e",
        "scinder": "#9467bd",
        "rejeter": "#d62728",
    }
    pct_rows: list[dict[str, Any]] = []
    for macro in macro_order:
        sub = verdict_df.loc[verdict_df["macro"] == macro]
        if sub.empty:
            continue
        counts = sub["verdict"].value_counts()
        total = float(len(sub))
        row: dict[str, Any] = {"macro": macro}
        for v in verdict_order:
            row[v] = 100.0 * float(counts.get(v, 0)) / total
        pct_rows.append(row)
    if not pct_rows:
        return out_paths

    pct_df = pd.DataFrame(pct_rows).set_index("macro").reindex(macro_order).fillna(0.0)
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    bottom = np.zeros(len(pct_df))
    x = np.arange(len(pct_df))
    for verdict in verdict_order:
        vals = pct_df[verdict].to_numpy()
        ax2.bar(
            x,
            vals,
            bottom=bottom,
            label=verdict,
            color=verdict_colors.get(verdict, "#999999"),
        )
        bottom = bottom + vals
    ax2.set_xticks(x)
    ax2.set_xticklabels(pct_df.index.tolist())
    ax2.set_ylabel("% topics")
    ax2.set_xlabel("Étape")
    ax2.set_title("Verdicts judge LLM par étape")
    ax2.legend(loc="upper right", fontsize=8, ncol=2)
    ax2.set_ylim(0, 100)
    plt.tight_layout()
    if fig_dir:
        verdict_png = Path(fig_dir) / "topic_judge_verdict_by_macro.png"
        fig2.savefig(verdict_png, dpi=140, bbox_inches="tight")
        out_paths["verdict_by_macro"] = verdict_png
    if show:
        plt.show()
    plt.close(fig2)
    return out_paths


def notebook_topic_judge_section_md() -> str:
    return TOPIC_JUDGE_SECTION_MD


def notebook_topic_judge_source(
    out_dir_var: str = "OUT_DIR",
    fig_dir_var: str = "FIG_DIR",
    *,
    restimate_var: str | None = "RESTIMATE",
    topic_judge_cfg_var: str | None = "TOPIC_JUDGE_CFG",
    seed: int = 42,
) -> str:
    """Code source réutilisable pour une section judge dans les notebooks."""
    restimate_expr = f"bool({restimate_var})" if restimate_var else "False"
    if topic_judge_cfg_var:
        cfg_expr = f"dict({topic_judge_cfg_var})"
    else:
        cfg_expr = (
            "dict(load_bertopic_macro_shared(anchor=TEXT_ROOT).get('topic_judge') or {})"
        )
    return f"""
from pathlib import Path

from macro_transfer.bertopic_config import load_bertopic_macro_shared
from macro_transfer.notebook_viz import load_topic_judge_artifacts, plot_topic_judge_quality
from macro_transfer.topic_judge import run_topic_judge_evaluation

_topic_judge_cfg = {cfg_expr}
_judge_scores_path = Path({out_dir_var}) / "summary" / "topic_judge_scores.csv"
_need_judge = {restimate_expr} or not _judge_scores_path.is_file()

if _topic_judge_cfg.get("enabled", False) and _need_judge:
    _themes_path = Path({out_dir_var}) / "topics_bertopic" / "themes_by_macro.csv"
    _assign_path = Path({out_dir_var}) / "topics_bertopic" / "assignments.csv"
    if not _themes_path.is_file() or not _assign_path.is_file():
        print("BERTopic absent — judge ignoré (themes/assignments manquants).")
    else:
        _judge_themes = pd.read_csv(_themes_path)
        _judge_assign = pd.read_csv(_assign_path)
        _meta_path = Path({out_dir_var}) / "transfer" / "bertopic_input_all.csv"
        if _meta_path.is_file():
            _judge_meta = pd.read_csv(_meta_path)
        else:
            _judge_meta = pd.DataFrame()
        _text_col = "sentence" if "sentence" in _judge_meta.columns else "text"
        if _judge_meta.empty or _text_col not in _judge_meta.columns:
            print("Meta texte absente — judge ignoré.")
        else:
            _judge_run = run_topic_judge_evaluation(
                Path({out_dir_var}),
                _judge_meta,
                _judge_assign,
                _judge_themes,
                cfg=_topic_judge_cfg,
                text_col=_text_col,
                seed={seed},
                force={restimate_expr},
            )
            print("LLM judge :", _judge_run)
elif not _topic_judge_cfg.get("enabled", False):
    print("topic_judge désactivé dans la config.")

_judge_art = load_topic_judge_artifacts(Path({out_dir_var}))
if _judge_art["scores"].empty:
    print("topic_judge_scores.csv absent — relancer le pipeline BERTopic ou activer RESTIMATE.")
else:
    _show_cols = [
        c for c in [
            "macro", "topic_id", "n_units", "label_propose", "score_global",
            "verdict", "probleme_principal", "justification_courte",
        ]
        if c in _judge_art["scores"].columns
    ]
    display(_judge_art["scores"][_show_cols].head(20))
    if not _judge_art["macro_summary"].empty:
        display(_judge_art["macro_summary"])

_judge_figs = plot_topic_judge_quality(Path({out_dir_var}), fig_dir=Path({fig_dir_var}), show=True)
if any(_judge_figs.values()):
    print("Figures judge :", {{k: str(v) for k, v in _judge_figs.items() if v}})
""".strip()


TOPIC_JUDGE_SECTION_MD = (
    "## Évaluation LLM-judge des topics BERTopic\n\n"
    "Chaque topic `(macro, topic_id)` est noté par un LLM sur **6 critères** (0–5) : "
    "cohérence interne, homogénéité accidentologique, alignement au rôle de l'étape, "
    "spécificité, nommabilité, utilité pour la reconstruction de scénario. "
    "Le **`score_global`** est la moyenne Python des 6 scores (non calculée par le LLM). "
    "Verdict : `conserver`, `fusionner`, `scinder`, `rejeter`.\n\n"
    "Sorties : `summary/topic_judge_scores.csv`, `summary/topic_judge_macro_summary.csv`."
)


def load_topic_judge_artifacts(out_dir: str | Path) -> Dict[str, Any]:
    """Charge scores judge + agrégats macro (DataFrames vides si absents)."""
    from macro_transfer.topic_judge import (
        load_topic_judge_macro_summary,
        load_topic_judge_scores,
        topic_judge_macro_summary_path,
        topic_judge_scores_path,
    )

    root = Path(out_dir).resolve()
    scores = load_topic_judge_scores(root)
    macro_summary = load_topic_judge_macro_summary(root)
    return {
        "scores": scores,
        "macro_summary": macro_summary,
        "scores_path": topic_judge_scores_path(root),
        "macro_summary_path": topic_judge_macro_summary_path(root),
    }


def plot_topic_judge_quality(
    out_dir: str | Path,
    *,
    fig_dir: Optional[Path] = None,
    score_threshold: float = 3.0,
    show: bool = True,
) -> Dict[str, Optional[Path]]:
    """
    Boxplot ``score_global`` par macro + barres empilées des verdicts.

    Écrit ``topic_judge_score_by_macro.png`` et ``topic_judge_verdict_by_macro.png``.
    """
    artifacts = load_topic_judge_artifacts(out_dir)
    scores = artifacts["scores"]
    out_paths: Dict[str, Optional[Path]] = {
        "score_by_macro": None,
        "verdict_by_macro": None,
    }
    if scores.empty or "score_global" not in scores.columns:
        print("topic_judge_scores.csv absent ou vide.")
        return out_paths

    df = scores.copy()
    df["macro"] = df["macro"].astype(str)
    df["score_global"] = pd.to_numeric(df["score_global"], errors="coerce")
    df = df.dropna(subset=["score_global"])
    if df.empty:
        print("Aucun score_global exploitable.")
        return out_paths

    macro_order = [m for m in MACRO_NAMES if m in set(df["macro"])]
    if not macro_order:
        macro_order = sorted(df["macro"].unique())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [df.loc[df["macro"] == m, "score_global"].to_numpy() for m in macro_order]
    ax.boxplot(data, labels=macro_order, patch_artist=True)
    ax.axhline(score_threshold, color="#d62728", linestyle="--", linewidth=1.2, label=f"seuil {score_threshold}")
    ax.set_title("Score global judge LLM par étape")
    ax.set_xlabel("Étape")
    ax.set_ylabel("score_global (0–5)")
    ax.set_ylim(0, 5.2)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    score_png = None
    if fig_dir:
        fig_dir_p = Path(fig_dir)
        fig_dir_p.mkdir(parents=True, exist_ok=True)
        score_png = fig_dir_p / "topic_judge_score_by_macro.png"
        fig.savefig(score_png, dpi=140, bbox_inches="tight")
        out_paths["score_by_macro"] = score_png
    if show:
        plt.show()
    plt.close(fig)

    verdict_df = df.copy()
    verdict_df["verdict"] = verdict_df["verdict"].astype(str).str.lower()
    verdict_order = ["conserver", "fusionner", "scinder", "rejeter"]
    verdict_colors = {
        "conserver": "#2ca02c",
        "fusionner": "#ff7f0e",
        "scinder": "#9467bd",
        "rejeter": "#d62728",
    }
    pct_rows: list[dict[str, Any]] = []
    for macro in macro_order:
        sub = verdict_df.loc[verdict_df["macro"] == macro]
        if sub.empty:
            continue
        counts = sub["verdict"].value_counts()
        total = float(len(sub))
        row: dict[str, Any] = {"macro": macro}
        for v in verdict_order:
            row[v] = 100.0 * float(counts.get(v, 0)) / total
        pct_rows.append(row)
    if not pct_rows:
        return out_paths

    pct_df = pd.DataFrame(pct_rows).set_index("macro").reindex(macro_order).fillna(0.0)
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    bottom = np.zeros(len(pct_df))
    x = np.arange(len(pct_df))
    for verdict in verdict_order:
        vals = pct_df[verdict].to_numpy()
        ax2.bar(
            x,
            vals,
            bottom=bottom,
            label=verdict,
            color=verdict_colors.get(verdict, "#999999"),
        )
        bottom = bottom + vals
    ax2.set_xticks(x)
    ax2.set_xticklabels(pct_df.index.tolist())
    ax2.set_ylabel("% topics")
    ax2.set_xlabel("Étape")
    ax2.set_title("Verdicts judge LLM par étape")
    ax2.legend(loc="upper right", fontsize=8, ncol=2)
    ax2.set_ylim(0, 100)
    plt.tight_layout()
    if fig_dir:
        verdict_png = Path(fig_dir) / "topic_judge_verdict_by_macro.png"
        fig2.savefig(verdict_png, dpi=140, bbox_inches="tight")
        out_paths["verdict_by_macro"] = verdict_png
    if show:
        plt.show()
    plt.close(fig2)
    return out_paths


def notebook_topic_judge_section_md() -> str:
    return TOPIC_JUDGE_SECTION_MD


def notebook_topic_judge_source(
    out_dir_var: str = "OUT_DIR",
    fig_dir_var: str = "FIG_DIR",
    *,
    restimate_var: str | None = "RESTIMATE",
    topic_judge_cfg_var: str | None = "TOPIC_JUDGE_CFG",
    seed: int = 42,
) -> str:
    """Code source réutilisable pour une section judge dans les notebooks."""
    restimate_expr = f"bool({restimate_var})" if restimate_var else "False"
    if topic_judge_cfg_var:
        cfg_expr = f"dict({topic_judge_cfg_var})"
    else:
        cfg_expr = (
            "dict(load_bertopic_macro_shared(anchor=TEXT_ROOT).get('topic_judge') or {})"
        )
    return f"""
from macro_transfer.bertopic_config import load_bertopic_macro_shared
from macro_transfer.notebook_viz import load_topic_judge_artifacts, plot_topic_judge_quality
from macro_transfer.topic_judge import run_topic_judge_evaluation

_topic_judge_cfg = {cfg_expr}
_judge_scores_path = Path({out_dir_var}) / "summary" / "topic_judge_scores.csv"
_need_judge = {restimate_expr} or not _judge_scores_path.is_file()

if _topic_judge_cfg.get("enabled", False) and _need_judge:
    _themes_path = Path({out_dir_var}) / "topics_bertopic" / "themes_by_macro.csv"
    _assign_path = Path({out_dir_var}) / "topics_bertopic" / "assignments.csv"
    if not _themes_path.is_file() or not _assign_path.is_file():
        print("BERTopic absent — judge ignoré (themes/assignments manquants).")
    else:
        _judge_themes = pd.read_csv(_themes_path)
        _judge_assign = pd.read_csv(_assign_path)
        _meta_path = Path({out_dir_var}) / "transfer" / "bertopic_input_all.csv"
        if _meta_path.is_file():
            _judge_meta = pd.read_csv(_meta_path)
        else:
            _judge_meta = pd.DataFrame()
        _text_col = "sentence" if "sentence" in _judge_meta.columns else "text"
        if _judge_meta.empty or _text_col not in _judge_meta.columns:
            print("Meta texte absente — judge ignoré.")
        else:
            _judge_run = run_topic_judge_evaluation(
                Path({out_dir_var}),
                _judge_meta,
                _judge_assign,
                _judge_themes,
                cfg=_topic_judge_cfg,
                text_col=_text_col,
                seed={seed},
                force={restimate_expr},
            )
            print("LLM judge :", _judge_run)
elif not _topic_judge_cfg.get("enabled", False):
    print("topic_judge désactivé dans la config.")

_judge_art = load_topic_judge_artifacts(Path({out_dir_var}))
if _judge_art["scores"].empty:
    print("topic_judge_scores.csv absent — relancer le pipeline BERTopic ou activer RESTIMATE.")
else:
    _show_cols = [
        c for c in [
            "macro", "topic_id", "n_units", "label_propose", "score_global",
            "verdict", "probleme_principal", "justification_courte",
        ]
        if c in _judge_art["scores"].columns
    ]
    display(_judge_art["scores"][_show_cols].head(20))
    if not _judge_art["macro_summary"].empty:
        display(_judge_art["macro_summary"])

_judge_figs = plot_topic_judge_quality(Path({out_dir_var}), fig_dir=Path({fig_dir_var}), show=True)
if any(_judge_figs.values()):
    print("Figures judge :", {{k: str(v) for k, v in _judge_figs.items() if v}})
""".strip()
