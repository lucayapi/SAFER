"""Visualisations notebook pour macro_transfer (UMAP, DataMapPlot, PCA/t-SNE)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from macro_transfer.constants import MACRO_NAMES


@dataclass
class RunArtifacts:
    """Chemins et tableaux d'un run macro_transfer."""

    out_dir: Path
    z: np.ndarray
    meta: pd.DataFrame
    gating: pd.DataFrame
    transfer_metrics: Dict[str, Any]
    topics_bertopic_dir: Path
    topics_gmm_dir: Path


def load_run_artifacts(out_dir: str | Path) -> RunArtifacts:
    """Charge projected.npy, metadata transfert et métriques."""
    root = Path(out_dir).resolve()
    emb = root / "embeddings"
    z_path = emb / "projected.npy"
    if not z_path.is_file():
        raise FileNotFoundError(f"Embeddings manquants : {z_path}")
    z = np.load(z_path)
    transfer_dir = root / "transfer"
    meta_path = transfer_dir / "metadata_with_macro_probs.csv"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Métadonnées transfert manquantes : {meta_path}")
    meta = pd.read_csv(meta_path)
    gating_cols = [c for c in meta.columns if c.startswith("p_") or c in ("m_hat", "q_conf", "ambiguous")]
    gating = meta[gating_cols].copy() if gating_cols else pd.DataFrame(index=meta.index)
    metrics: Dict[str, Any] = {}
    mpath = transfer_dir / "transfer_metrics.json"
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
        topics_gmm_dir=root / "topics_gmm",
    )


def _theme_label_map(themes: pd.DataFrame) -> Dict[Tuple[str, int], str]:
    if themes.empty or "macro" not in themes.columns or "topic_id" not in themes.columns:
        return {}
    out: Dict[Tuple[str, int], str] = {}
    for _, row in themes.iterrows():
        macro = str(row["macro"])
        tid = int(row["topic_id"])
        words = str(row.get("top_words", row.get("theme_summary", "")))[:80]
        out[(macro, tid)] = f"{macro}|T{tid}: {words}" if words else f"{macro}|T{tid}"
    return out


def merge_assignments(
    meta: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    confidence_threshold: float = 0.5,
    macro_col: str = "m_hat",
) -> pd.DataFrame:
    """Joint meta + gating + topic_id (filtrage q_conf)."""
    df = meta.copy()
    if macro_col not in df.columns and "m_hat" in df.columns:
        macro_col = "m_hat"
    if "q_conf" in df.columns:
        df = df.loc[df["q_conf"].astype(float) >= confidence_threshold].copy()
    if assignments.empty or "doc_idx" not in assignments.columns:
        df["topic_id"] = -1
        return df.reset_index(drop=True)
    sub = assignments.loc[assignments["topic_id"] >= 0, ["doc_idx", "macro", "topic_id"]].copy()
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
    """UMAP / DataMapPlot / PCA-t-SNE pour une macro et un algo (BERTopic ou GMM)."""
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
    tmap = _theme_label_map(themes.loc[themes["macro"].astype(str) == macro])

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
        ax.scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=0.6, c=[pal[i]], label=f"T{tid}")
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
                ax.scatter(xy[mask, 0], xy[mask, 1], s=10, alpha=0.6, c=[pal[i]], label=f"T{tid}")
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
