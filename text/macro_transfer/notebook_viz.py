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

# Couleurs fixes par macro (lisibles sur fond pastel).
MACRO_COLOR_HEX: Dict[str, str] = {
    "A0": "#1f77b4",
    "A1": "#ff7f0e",
    "B": "#2ca02c",
    "C": "#d62728",
}


@dataclass
class RunArtifacts:
    """Chemins et tableaux d'un run macro_transfer."""

    out_dir: Path
    z: np.ndarray
    meta: pd.DataFrame
    gating: pd.DataFrame
    transfer_metrics: Dict[str, Any]
    topics_bertopic_dir: Path


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


def load_run_artifacts(out_dir: str | Path) -> RunArtifacts:
    """Charge embeddings TPN adaptés, metadata gating et métriques adaptées."""
    root = Path(out_dir).resolve()
    emb = root / "embeddings"
    z_path = emb / "target_adapted.npy"
    if not z_path.is_file():
        z_path = emb / "target_projected.npy"
    if not z_path.is_file():
        raise FileNotFoundError(
            f"Embeddings TPN manquants sous {emb} "
            "(attendu target_adapted.npy ou target_projected.npy)."
        )
    z = np.load(z_path)
    transfer_dir = root / "transfer"
    meta_path = transfer_dir / "metadata_with_tpn_macro_probs.csv"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Métadonnées TPN manquantes : {meta_path}")
    meta = pd.read_csv(meta_path)
    gating_cols = [c for c in meta.columns if c.startswith("p_") or c in ("m_hat", "q_conf", "ambiguous")]
    gating = meta[gating_cols].copy() if gating_cols else pd.DataFrame(index=meta.index)
    metrics: Dict[str, Any] = {}
    mpath = transfer_dir / "transfer_metrics_adapted.json"
    if mpath.is_file():
        with open(mpath, encoding="utf-8") as f:
            metrics = json.load(f)
    return RunArtifacts(
        out_dir=root,
        z=z,
        meta=meta,
        gating=gating,
        transfer_metrics=metrics,
        topics_bertopic_dir=root / "topics_bertopic",
    )


def _read_optional_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    return pd.read_csv(path)


def load_fsp_run_artifacts(out_dir: str | Path) -> FSPRunArtifacts:
    """Charge les sorties baseline Frozen Source Prototypes."""
    root = Path(out_dir).resolve()
    transfer = root / "transfer"
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
    ax.set_title("Répartition des macros prédites")
    ax.set_xlabel("pred_macro")
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
    ax.set_title("Distances aux prototypes par macro prédite")
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


def prepare_topic_labeled_points(
    artifacts: RunArtifacts,
    *,
    topic_subdir: str = "topics_bertopic",
    confidence_threshold: float = 0.5,
    max_points: int = 8000,
    seed: int = 42,
    label_max_chars: int = 52,
) -> Optional[Tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    """
    Sous-ensemble (z, libellés topic) pour cartes globales.

    Ne garde que les unités avec topic_id ≥ 0 et q_conf ≥ seuil.
    """
    root = artifacts.out_dir / topic_subdir
    themes_path = root / "themes_by_macro.csv"
    assign_path = root / "assignments.csv"
    if not themes_path.is_file() or not assign_path.is_file():
        print(f"[topics] absent : {themes_path} ou {assign_path}")
        return None

    themes = pd.read_csv(themes_path)
    assignments = pd.read_csv(assign_path)
    merged = merge_assignments(
        artifacts.meta,
        assignments,
        confidence_threshold=confidence_threshold,
    )
    merged = merged.loc[merged["topic_id"] >= 0].copy()
    if len(merged) < 5:
        print("[topics] trop peu de points assignés (topic_id ≥ 0)")
        return None

    if "macro" in merged.columns:
        macro_col = "macro"
    elif "macro_y" in merged.columns:
        macro_col = "macro_y"
    else:
        macro_col = "m_hat"
    tmap = _theme_label_map(themes, max_chars=label_max_chars)
    idx_rows = merged["doc_idx"].astype(int).to_numpy()
    z_sub = artifacts.z[idx_rows]
    dm_labels = np.array(
        [
            tmap.get((str(row[macro_col]), int(row["topic_id"])), f"{row[macro_col]}·T{int(row['topic_id'])}")
            for _, row in merged.iterrows()
        ],
        dtype=object,
    )

    n = min(max_points, len(z_sub))
    rng = np.random.default_rng(seed)
    if len(z_sub) > n:
        pick = rng.choice(len(z_sub), size=n, replace=False)
        z_sub = z_sub[pick]
        dm_labels = dm_labels[pick]
        merged = merged.iloc[pick].reset_index(drop=True)

    return z_sub, dm_labels, merged


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


def plot_transfer_macro_overview(
    artifacts: RunArtifacts,
    *,
    label_col: str = "pred_label",
    confidence_threshold: float = 0.5,
    fig_dir: Optional[Path] = None,
) -> None:
    """Histogramme q_conf + barres accuracy si labels présents."""
    meta = artifacts.meta
    fig_dir = Path(fig_dir) if fig_dir else None

    if "q_conf" in meta.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(meta["q_conf"].astype(float).dropna(), bins=40, color="steelblue", edgecolor="white")
        ax.axvline(confidence_threshold, color="crimson", ls="--", label=f"seuil {confidence_threshold}")
        ax.set_title("Confiance macro (max p(m|u))")
        ax.set_xlabel("q_conf")
        ax.legend()
        plt.tight_layout()
        if fig_dir:
            fig.savefig(fig_dir / "hist_q_conf.png", dpi=140, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    m = artifacts.transfer_metrics
    if m:
        print("=== Métriques transfert ===")
        display_df = pd.DataFrame([m])
        try:
            from IPython.display import display

            display(display_df)
        except ImportError:
            print(display_df.to_string())

    if label_col in meta.columns and "m_hat" in meta.columns:
        ct = pd.crosstab(meta[label_col].astype(str), meta["m_hat"].astype(str))
        fig, ax = plt.subplots(figsize=(6, 5))
        import seaborn as sns

        sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Vérité (lignes) vs macro prédite (colonnes)")
        plt.tight_layout()
        if fig_dir:
            fig.savefig(fig_dir / "confusion_pred_vs_mhat.png", dpi=140, bbox_inches="tight")
        plt.show()
        plt.close(fig)


def _topic_centroids_2d(
    coords: np.ndarray,
    labels: np.ndarray,
) -> Tuple[List[float], List[float], List[str]]:
    """Centroïde 2D par libellé topic (pour annotations matplotlib)."""
    labels = np.asarray(labels).astype(str)
    cx, cy, names = [], [], []
    for name in sorted(set(labels)):
        mask = labels == name
        if not mask.any():
            continue
        mu = coords[mask].mean(axis=0)
        cx.append(float(mu[0]))
        cy.append(float(mu[1]))
        names.append(name)
    return cx, cy, names


def plot_global_topics_datamap(
    artifacts: RunArtifacts,
    *,
    algo_tag: str,
    confidence_threshold: float = 0.5,
    max_points: int = 8000,
    seed: int = 42,
    fig_dir: Optional[Path] = None,
    use_datamap: bool = True,
    label_font_size: int = 7,
    label_max_chars: int = 52,
) -> None:
    """UMAP + DataMapPlot global : une étiquette par topic (theme_label), pas seulement la macro."""
    from scgm_text.notebook_viz import plot_umap_datamap_static

    prep = prepare_topic_labeled_points(
        artifacts,
        confidence_threshold=confidence_threshold,
        max_points=max_points,
        seed=seed,
        label_max_chars=label_max_chars,
    )
    if prep is None:
        return
    z_sub, dm_labels, _merged = prep
    coords = _run_umap(z_sub, random_state=seed)
    fig_dir = Path(fig_dir) if fig_dir else None
    safe = str(algo_tag).replace(" ", "_").lower()

    if use_datamap:
        try:
            fig, _ = plot_umap_datamap_static(
                coords,
                dm_labels,
                title=f"{algo_tag} — topics BERTopic (libellés theme_label)",
                label_font_size=label_font_size,
                macro_centroids=None,
            )
            if fig_dir:
                out = fig_dir / f"global_topics_datamap_{safe}.png"
                fig.savefig(out, dpi=150, bbox_inches="tight")
                print("Figure :", out)
            plt.show()
            plt.close(fig)
        except Exception as exc:
            print("DataMapPlot ignoré :", exc)

    import seaborn as sns

    fig, ax = plt.subplots(figsize=(12, 8))
    uniq = sorted(set(dm_labels), key=str)
    pal = sns.color_palette("husl", n_colors=max(len(uniq), 1))
    for i, lab in enumerate(uniq):
        mask = dm_labels == lab
        ax.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.45, c=[pal[i]])
    tcx, tcy, tnames = _topic_centroids_2d(coords, dm_labels)
    for xi, yi, name in zip(tcx, tcy, tnames):
        ax.annotate(
            name,
            (xi, yi),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6,
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"),
        )
    ax.set_title(f"UMAP — {algo_tag} (topics, annotations centroïdes)")
    plt.tight_layout()
    if fig_dir:
        fig.savefig(fig_dir / f"global_topics_scatter_{safe}.png", dpi=140, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_global_topics_compare_methods(
    artifacts_by_method: Dict[str, RunArtifacts],
    *,
    confidence_threshold: float = 0.5,
    max_points: int = 6000,
    seed: int = 42,
    fig_dir: Optional[Path] = None,
    use_datamap: bool = True,
    label_font_size: int = 7,
) -> None:
    """SCGM vs SoftTriple côte à côte (DataMapPlot + scatter annoté par topic)."""
    from scgm_text.notebook_viz import plot_umap_datamap_static

    methods = list(artifacts_by_method.items())
    if not methods:
        return
    fig_dir = Path(fig_dir) if fig_dir else None

    if use_datamap:
        for tag, art in methods:
            print(f"=== DataMapPlot topics — {tag} ===")
            plot_global_topics_datamap(
                art,
                algo_tag=tag,
                confidence_threshold=confidence_threshold,
                max_points=max_points,
                seed=seed,
                fig_dir=fig_dir / tag.lower().replace(" ", "_") if fig_dir else None,
                use_datamap=True,
                label_font_size=label_font_size,
            )

    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 8), squeeze=False)
    import seaborn as sns

    for ax, (tag, art) in zip(axes[0], methods):
        prep = prepare_topic_labeled_points(
            art,
            confidence_threshold=confidence_threshold,
            max_points=max_points,
            seed=seed,
        )
        if prep is None:
            ax.set_title(f"{tag} — (données absentes)")
            ax.axis("off")
            continue
        z_sub, dm_labels, _ = prep
        coords = _run_umap(z_sub, random_state=seed)
        uniq = sorted(set(dm_labels), key=str)
        pal = sns.color_palette("husl", n_colors=max(len(uniq), 1))
        for i, lab in enumerate(uniq):
            mask = dm_labels == lab
            ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.4, c=[pal[i]])
        tcx, tcy, tnames = _topic_centroids_2d(coords, dm_labels)
        for xi, yi, name in zip(tcx, tcy, tnames):
            short = name if len(name) <= 40 else name[:39] + "…"
            ax.annotate(short, (xi, yi), fontsize=5, alpha=0.9)
        ax.set_title(f"{tag} — topics (centroïdes)")
    plt.tight_layout()
    if fig_dir:
        out = fig_dir / "compare_scgm_softtriple_topics.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("Figure comparative :", out)
    plt.show()
    plt.close(fig)


def _run_umap(X: np.ndarray, *, random_state: int = 42) -> np.ndarray:
    from umap import UMAP

    n = len(X)
    n_neighbors = min(15, max(2, n - 1))
    reducer = UMAP(
        n_components=2,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
    )
    return reducer.fit_transform(X)


def plot_global_embedding_map(
    artifacts: RunArtifacts,
    *,
    max_points: int = 8000,
    confidence_threshold: float = 0.5,
    seed: int = 42,
    fig_dir: Optional[Path] = None,
    use_datamap: bool = True,
) -> None:
    """UMAP global coloré par m_hat ; option DataMapPlot + Plotly."""
    from scgm_text.notebook_viz import (
        display_plotly_html,
        macro_umap_centroids,
        plot_umap_datamap_static,
        plot_umap_plotly,
        sample_projection_indices,
    )

    meta = artifacts.meta
    z = artifacts.z
    if "m_hat" not in meta.columns:
        print("(absent) m_hat — pas de carte globale")
        return

    label_col = "m_hat"
    idx = sample_projection_indices(meta, label_col, max_points=max_points, seed=seed)
    X = z[idx]
    lab_df = meta.iloc[idx].copy()
    coords = _run_umap(X, random_state=seed)
    labels = lab_df[label_col].astype(str).to_numpy()

    fig_dir = Path(fig_dir) if fig_dir else None
    centroids = macro_umap_centroids(coords, labels)

    if use_datamap:
        try:
            fig, _ = plot_umap_datamap_static(
                coords,
                labels,
                title="Corpus test — macro (m_hat)",
                label_font_size=7,
                macro_centroids=centroids,
            )
            if fig_dir:
                out = fig_dir / "global_umap_datamap_macro.png"
                fig.savefig(out, dpi=150, bbox_inches="tight")
                print("Figure :", out)
            plt.show()
            plt.close(fig)
        except Exception as exc:
            print("DataMapPlot ignoré :", exc)

    fig_pl = plot_umap_plotly(
        coords,
        lab_df,
        label_col=label_col,
        hover_label=label_col,
        title="UMAP global — macro (interactif)",
        out_html=fig_dir / "global_umap_interactive.html" if fig_dir else None,
    )
    if fig_dir and fig_pl is not None:
        display_plotly_html(fig_dir / "global_umap_interactive.html")

    fig, ax = plt.subplots(figsize=(8, 6))
    import seaborn as sns

    pal = sns.color_palette("Set2", n_colors=len(MACRO_NAMES))
    for i, m in enumerate(MACRO_NAMES):
        mask = labels == m
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.5, c=[pal[i]], label=m)
    ax.set_title("UMAP — couleur macro (m_hat)")
    ax.legend(markerscale=2)
    plt.tight_layout()
    if fig_dir:
        fig.savefig(fig_dir / "global_umap_scatter_macro.png", dpi=140, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_topics_per_macro(
    artifacts: RunArtifacts,
    *,
    topic_subdir: str,
    algo_tag: str,
    macro: str,
    confidence_threshold: float = 0.5,
    max_points: int = 4000,
    seed: int = 42,
    fig_dir: Optional[Path] = None,
    use_datamap: bool = True,
    run_pca_tsne: bool = True,
) -> None:
    """UMAP / DataMapPlot / PCA-t-SNE pour une macro (topics BERTopic)."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    from scgm_text.notebook_viz import plot_umap_datamap_static

    root = artifacts.out_dir / topic_subdir
    themes_path = root / "themes_by_macro.csv"
    assign_path = root / "assignments.csv"
    if not themes_path.is_file() or not assign_path.is_file():
        print(f"[{algo_tag}] absent sous {root}")
        return

    themes = pd.read_csv(themes_path)
    assignments = pd.read_csv(assign_path)
    macro_assign = assignments.loc[assignments["macro"].astype(str) == macro].copy()
    merged = merge_assignments(
        artifacts.meta,
        macro_assign,
        confidence_threshold=confidence_threshold,
    )
    merged = merged.loc[merged["m_hat"].astype(str) == macro]
    merged = merged.loc[merged["topic_id"] >= 0]
    if len(merged) < 5:
        print(f"[{algo_tag} / {macro}] trop peu de points après filtre")
        return

    idx_rows = merged["doc_idx"].astype(int).to_numpy()
    z_sub = artifacts.z[idx_rows]
    label_topics = merged["topic_id"].astype(str).to_numpy()
    tmap = _theme_label_map(themes.loc[themes["macro"].astype(str) == macro], max_chars=52)

    n = min(max_points, len(z_sub))
    rng = np.random.default_rng(seed)
    if len(z_sub) > n:
        pick = rng.choice(len(z_sub), size=n, replace=False)
        z_sub = z_sub[pick]
        label_topics = label_topics[pick]
        merged_sub = merged.iloc[pick]
    else:
        merged_sub = merged

    coords = _run_umap(z_sub, random_state=seed)
    dm_labels = np.array(
        [tmap.get((macro, int(tid)), f"T{tid}") for tid in label_topics],
        dtype=object,
    )

    fig_dir = Path(fig_dir) if fig_dir else None
    safe = f"{algo_tag}_{macro}".replace(" ", "_")

    if use_datamap:
        try:
            fig, _ = plot_umap_datamap_static(
                coords,
                dm_labels,
                title=f"{algo_tag} — {macro} (topics intra-macro)",
                label_font_size=6,
            )
            if fig_dir:
                p = fig_dir / f"umap_datamap_{safe}.png"
                fig.savefig(p, dpi=150, bbox_inches="tight")
                print("Figure :", p)
            plt.show()
            plt.close(fig)
        except Exception as exc:
            print("DataMapPlot :", exc)

    fig, ax = plt.subplots(figsize=(8, 6))
    import seaborn as sns

    topics_unique = sorted(set(label_topics), key=lambda x: int(x) if str(x).lstrip("-").isdigit() else 0)
    pal = sns.color_palette("husl", n_colors=max(len(topics_unique), 1))
    for i, tid in enumerate(topics_unique):
        mask = label_topics == tid
        leg = tmap.get((macro, int(tid)), f"T{tid}")
        ax.scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=0.6, c=[pal[i]], label=leg)
    ax.set_title(f"UMAP — {algo_tag} / {macro}")
    ax.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    if fig_dir:
        fig.savefig(fig_dir / f"umap_scatter_{safe}.png", dpi=140, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    if run_pca_tsne and len(z_sub) >= 10:
        pca_xy = PCA(n_components=2, random_state=seed).fit_transform(z_sub)
        tsne_xy = TSNE(
            n_components=2,
            random_state=seed,
            init="pca",
            learning_rate="auto",
        ).fit_transform(z_sub)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, xy, title in (
            (axes[0], pca_xy, f"PCA — {algo_tag} {macro}"),
            (axes[1], tsne_xy, f"t-SNE — {algo_tag} {macro}"),
        ):
            for i, tid in enumerate(topics_unique):
                mask = label_topics == tid
                leg = tmap.get((macro, int(tid)), f"T{tid}")
                ax.scatter(xy[mask, 0], xy[mask, 1], s=10, alpha=0.6, c=[pal[i]], label=leg)
            ax.set_title(title)
            ax.legend(markerscale=2, fontsize=7)
        plt.tight_layout()
        if fig_dir:
            fig.savefig(fig_dir / f"pca_tsne_{safe}.png", dpi=140, bbox_inches="tight")
        plt.show()
        plt.close(fig)


def display_topics_tables(out_dir: Path, topic_subdir: str, algo_tag: str) -> None:
    """Affiche themes_by_macro et effectifs."""
    root = Path(out_dir) / topic_subdir
    p = root / "themes_by_macro.csv"
    if not p.is_file():
        print(f"[{algo_tag}] absent : {p}")
        return
    th = pd.read_csv(p)
    print(f"=== {algo_tag} — themes_by_macro ===")
    try:
        from IPython.display import display

        display(th.sort_values(["macro", "n_units"], ascending=[True, False]))
        print("Effectifs par macro :")
        display(th.groupby("macro")["n_units"].sum().reset_index())
    except ImportError:
        print(th.head(20).to_string())


def _confusion_matrix_from_metrics(metrics: Dict[str, Any]) -> Optional[pd.DataFrame]:
    conf = metrics.get("confusion") or {}
    if not conf:
        return None
    rows = []
    for true_m in MACRO_NAMES:
        row = {"true": true_m}
        for pred_m in MACRO_NAMES:
            row[pred_m] = int(conf.get(true_m, {}).get(pred_m, 0))
        rows.append(row)
    cm = pd.DataFrame(rows).set_index("true")
    return cm.reindex(index=MACRO_NAMES, columns=MACRO_NAMES, fill_value=0)


def plot_confusion_from_metrics(
    metrics: Dict[str, Any],
    title: str,
    save_path: Optional[Path] = None,
    *,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Heatmap matrice de confusion depuis transfer_metrics_*.json."""
    cm = _confusion_matrix_from_metrics(metrics)
    if cm is None or cm.values.sum() == 0:
        print("Confusion absente ou vide —", title)
        return None
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Prédit (m_hat)")
    ax.set_ylabel("Vérité (pred_label)")
    ax.set_title(title)
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _subsample_stratified(
    n_source: int,
    n_target: int,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Indices source et cible sous-échantillonnés."""
    rng = np.random.default_rng(seed)
    half = max(2, max_points // 2)
    n_s = min(n_source, half)
    n_t = min(n_target, max_points - n_s)
    idx_s = (
        rng.choice(n_source, size=n_s, replace=False)
        if n_source > n_s
        else np.arange(n_source)
    )
    idx_t = (
        rng.choice(n_target, size=n_t, replace=False)
        if n_target > n_t
        else np.arange(n_target)
    )
    return np.asarray(idx_s, dtype=np.int64), np.asarray(idx_t, dtype=np.int64)


def _load_domain_embedding_pair(
    out_dir: Path,
    phase: str,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """phase: 'projected' (initial) ou 'adapted'."""
    emb = Path(out_dir) / "embeddings"
    if phase == "projected":
        p_s, p_t = emb / "source_projected.npy", emb / "target_projected.npy"
    else:
        p_s, p_t = emb / "source_adapted.npy", emb / "target_adapted.npy"
    if not p_s.is_file() or not p_t.is_file():
        return None, None
    return np.load(p_s), np.load(p_t)


def _tsne_2d(z: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.manifold import TSNE

    n = len(z)
    if n < 3:
        return np.zeros((n, 2), dtype=np.float64)
    perplexity = float(min(30, max(2, n - 1)))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    ).fit_transform(z)


def _plot_domain_panel(
    ax: plt.Axes,
    z_source: np.ndarray,
    z_target: np.ndarray,
    title: str,
    *,
    source_label: str = "Source BTP",
    target_label: str = "Test",
) -> None:
    z = np.vstack([z_source, z_target])
    xy = _tsne_2d(z, seed=42)
    n_s = len(z_source)
    ax.scatter(
        xy[:n_s, 0],
        xy[:n_s, 1],
        s=8,
        alpha=0.45,
        c="#1f77b4",
        label=source_label,
    )
    ax.scatter(
        xy[n_s:, 0],
        xy[n_s:, 1],
        s=8,
        alpha=0.45,
        c="#888888",
        label=target_label,
    )
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=8)


def plot_domain_tsne_side_by_side(
    out_dir: Path,
    fig_dir: Optional[Path] = None,
    *,
    max_points: int = 4000,
    seed: int = 42,
    source_label: str = "Source BTP",
    target_label: str = "Test",
    test_corpus_name: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Figure 1×2 : t-SNE source (bleu) + cible (gris) — encodage initial (gauche) vs adapté (droite).
    """
    root = Path(out_dir)
    z_s_proj, z_t_proj = _load_domain_embedding_pair(root, "projected")
    z_s_adapt, z_t_adapt = _load_domain_embedding_pair(root, "adapted")
    if z_s_proj is None or z_t_proj is None:
        print(f"Embeddings domaine manquants sous {root / 'embeddings'}")
        return None

    n_source = len(z_s_proj)
    n_target = len(z_t_proj)
    manifest_path = root / "run_manifest.json"
    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            man = json.load(f)
        n_source = int(man.get("n_source", n_source))
        n_target = int(man.get("n_target", n_target))

    idx_s, idx_t = _subsample_stratified(n_source, n_target, max_points, seed)
    tgt_name = target_label if test_corpus_name is None else f"Test {test_corpus_name}"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    _plot_domain_panel(
        axes[0],
        z_s_proj[idx_s],
        z_t_proj[idx_t],
        "Encodage initial (projeté)",
        source_label=source_label,
        target_label=tgt_name,
    )
    if z_s_adapt is not None and z_t_adapt is not None:
        _plot_domain_panel(
            axes[1],
            z_s_adapt[idx_s],
            z_t_adapt[idx_t],
            "Encodage adapté (TPN)",
            source_label=source_label,
            target_label=tgt_name,
        )
    else:
        axes[1].set_visible(False)
    fig.suptitle("t-SNE — domaines source vs test", y=1.02)
    plt.tight_layout()
    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_dir / "tsne_domain_initial_vs_adapted.png", dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _resolve_macro_column(df: pd.DataFrame) -> str:
    for col in ("macro", "m_hat", "macro_y", "pred_label"):
        if col in df.columns:
            return col
    return "m_hat"


def build_topics_display_dataframe(
    artifacts: RunArtifacts,
    *,
    confidence_threshold: float = 0.0,
    filter_meta_by_confidence: bool = False,
    topic_subdir: str = "topics_bertopic",
) -> pd.DataFrame:
    """
    Tableau unitaire prêt pour l'affichage texte coloré (toutes les phrases du corpus test).

    Joint meta TPN + ``assignments.csv`` (left join). Ajoute ``theme_label`` (long),
    ``theme_label_short`` (légende) et ``Probability`` (``prob`` ou ``p_mk``).
    """
    root = artifacts.out_dir / topic_subdir
    themes_path = root / "themes_by_macro.csv"
    assign_path = root / "assignments.csv"
    if not assign_path.is_file():
        raise FileNotFoundError(f"Assignations BERTopic manquantes : {assign_path}")

    assignments = pd.read_csv(assign_path)
    merged = merge_assignments(
        artifacts.meta,
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
    artifacts: RunArtifacts,
    accident_id: Any,
    *,
    confidence_threshold: float = 0.0,
    **kwargs: Any,
) -> None:
    """Charge meta+topics depuis un run TPN et affiche le récit coloré."""
    df = build_topics_display_dataframe(artifacts, confidence_threshold=confidence_threshold)
    show_colored_text_inline(df, accident_id, **kwargs)
