"""Visualisations notebook SCGM (matplotlib statique + Plotly interactif)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def _macro_palette() -> Dict[str, str]:
    import seaborn as sns

    macros = ["A0", "A1", "B", "C"]
    colors = sns.color_palette("Set2", 4)
    return {m: f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for m, c in zip(macros, colors)}


def resolve_datamap_labels(
    lab: pd.DataFrame,
    *,
    label_col: str,
    label_mode: str,
    themes_openai_path: Optional[PathLike],
) -> Tuple[np.ndarray, str]:
    """Construit les libellés DataMap (theme_summary ou macro|z)."""
    mode = str(label_mode).strip().lower()
    themes_path = Path(themes_openai_path) if themes_openai_path else None

    if mode == "theme_summary" and themes_path and themes_path.is_file():
        themes_df = pd.read_csv(themes_path)
        if "theme_summary" in themes_df.columns and "z_id" in themes_df.columns:
            z2s = dict(zip(themes_df["z_id"].astype(int), themes_df["theme_summary"].astype(str)))
            labels = lab["z_hat"].map(lambda z: z2s.get(int(z), f"z={int(z)}")).to_numpy(dtype=object)
            return labels, "theme_summary"
        print("themes_by_z_openai.csv : colonnes theme_summary ou z_id absentes ; repli macro_z.")
    elif mode == "theme_summary":
        print(
            f"Fichier absent : {themes_path} — exécuter la cellule OpenAI (11 bis) "
            "ou définir DATAMAP_LABEL_MODE='macro_z'. Repli macro|z."
        )

    labels = (lab[label_col].astype(str) + "|z=" + lab["z_hat"].astype(str)).to_numpy(dtype=object)
    return labels, "macro_z"


def sample_projection_indices(
    meta: pd.DataFrame,
    label_col: str,
    *,
    max_points: int,
    seed: int,
) -> np.ndarray:
    """Sous-échantillon stratifié par macro pour PCA / t-SNE."""
    per_macro = max(1, max_points // 4)
    sample_df = meta.groupby(label_col, group_keys=False).apply(
        lambda g: g.sample(min(len(g), per_macro), random_state=seed)
    )
    if len(sample_df) > max_points:
        sample_df = sample_df.sample(max_points, random_state=seed)
    return sample_df.index.to_numpy()


def plot_umap_datamap_static(
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    label_font_size: int = 8,
    macro_centroids: Optional[Tuple[List[float], List[float], List[str], Dict[str, str]]] = None,
):
    """Carte DataMapPlot (matplotlib). Retourne (fig, ax)."""
    import datamapplot as dmp

    fig, ax = dmp.create_plot(coords, labels, title=title, label_font_size=label_font_size)
    if macro_centroids is None:
        return fig, ax

    from matplotlib.lines import Line2D

    cx, cy, names, macro_to_color = macro_centroids
    if not names:
        return fig, ax

    ax.scatter(
        cx,
        cy,
        s=240,
        c=[macro_to_color[m] for m in names],
        marker="P",
        edgecolors="#111111",
        linewidths=1.2,
        zorder=100,
    )
    for xi, yi, m in zip(cx, cy, names):
        ax.annotate(
            m,
            (xi, yi),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="#111111",
            zorder=101,
        )
    leg_handles = [
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="P",
            color="w",
            markerfacecolor=macro_to_color[m],
            markeredgecolor="#111111",
            markersize=11,
            label=m,
        )
        for m in names
    ]
    ax.legend(handles=leg_handles, loc="lower left", frameon=True, title="Macro (pred_label)", fontsize=8, title_fontsize=9)
    return fig, ax


def _macro_color_map(macros_order: Sequence[str] = ("A0", "A1", "B", "C")) -> Dict[str, str]:
    from matplotlib import colors as mcolors

    try:
        import seaborn as sns

        pal = sns.color_palette("Set2", len(macros_order))
    except ImportError:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("Set2")
        pal = [cmap(i / max(len(macros_order) - 1, 1)) for i in range(len(macros_order))]
    pal_hex = [mcolors.to_hex(c) for c in pal]
    return dict(zip(macros_order, pal_hex))


def macro_centroids_2d(
    coords: np.ndarray,
    macro_labels: np.ndarray,
    macros_order: Sequence[str] = ("A0", "A1", "B", "C"),
) -> Tuple[List[float], List[float], List[str], Dict[str, str]]:
    """Moyenne 2D des points par macro (PCA, t-SNE ou UMAP)."""
    labels = np.asarray(macro_labels).astype(str)
    macro_to_color = _macro_color_map(macros_order)
    cx, cy, names = [], [], []
    for m in macros_order:
        mask = labels == m
        if not np.any(mask):
            continue
        mu = coords[mask].mean(axis=0)
        cx.append(float(mu[0]))
        cy.append(float(mu[1]))
        names.append(m)
    return cx, cy, names, macro_to_color


def macro_umap_centroids(
    coords: np.ndarray,
    macro_labels: np.ndarray,
    macros_order: Sequence[str] = ("A0", "A1", "B", "C"),
) -> Tuple[List[float], List[float], List[str], Dict[str, str]]:
    """Centroïdes UMAP par macro pour overlay matplotlib."""
    return macro_centroids_2d(coords, macro_labels, macros_order=macros_order)


def z_to_macro_map(themes_z: Optional[pd.DataFrame]) -> Dict[int, str]:
    if themes_z is None or "z_id" not in themes_z.columns:
        return {}
    macro_col = "dominant_macro" if "dominant_macro" in themes_z.columns else None
    if macro_col is None:
        return {}
    out: Dict[int, str] = {}
    for _, row in themes_z.iterrows():
        z_id = int(row["z_id"])
        macro = str(row.get(macro_col, "")).strip()
        if macro:
            out[z_id] = macro
    return out


def z_centroids_2d(
    coords: np.ndarray,
    z_ids: np.ndarray,
    z_to_macro: Dict[int, str],
    macros_order: Sequence[str] = ("A0", "A1", "B", "C"),
) -> Tuple[List[float], List[float], List[int], List[str]]:
    """Moyenne 2D des points par composante z ; couleur = macro dominante du topic."""
    macro_to_color = _macro_color_map(macros_order)
    z_arr = np.asarray(z_ids)
    cx, cy, z_list, colors = [], [], [], []
    for z in sorted(np.unique(z_arr)):
        mask = z_arr == z
        if not np.any(mask):
            continue
        mu = coords[mask].mean(axis=0)
        cx.append(float(mu[0]))
        cy.append(float(mu[1]))
        z_list.append(int(z))
        macro = z_to_macro.get(int(z), "")
        colors.append(macro_to_color.get(macro, "#888888"))
    return cx, cy, z_list, colors


def overlay_projection_centroids(
    ax,
    coords: np.ndarray,
    sample_df: pd.DataFrame,
    label_col: str,
    *,
    z_col: str = "z_hat",
    themes_z: Optional[pd.DataFrame] = None,
    show_macro: bool = True,
    show_z: bool = False,
) -> None:
    """Superpose centroïdes macro (X) et composantes z (*) sur un axe PCA/t-SNE."""
    from matplotlib.lines import Line2D

    n = len(sample_df)
    if coords.shape[0] != n:
        raise ValueError(f"coords ({coords.shape[0]}) vs sample_df ({n})")

    macro_labels = sample_df[label_col].astype(str).to_numpy()
    legend_handles: List[Line2D] = []

    if show_macro:
        cx, cy, names, macro_to_color = macro_centroids_2d(coords, macro_labels)
        if names:
            ax.scatter(
                cx,
                cy,
                s=140,
                c=[macro_to_color[m] for m in names],
                marker="X",
                edgecolors="#111111",
                linewidths=1.0,
                zorder=90,
            )
            for xi, yi, m in zip(cx, cy, names):
                ax.annotate(
                    m,
                    (xi, yi),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    color="#111111",
                    zorder=91,
                )
            legend_handles.extend(
                Line2D(
                    [0],
                    [0],
                    linestyle="None",
                    marker="X",
                    color="w",
                    markerfacecolor=macro_to_color[m],
                    markeredgecolor="#111111",
                    markersize=9,
                    label=m,
                )
                for m in names
            )

    if show_z and z_col in sample_df.columns:
        z_ids = sample_df[z_col].astype(int).to_numpy()
        z_map = z_to_macro_map(themes_z)
        zx, zy, z_list, z_colors = z_centroids_2d(coords, z_ids, z_map)
        if z_list:
            ax.scatter(
                zx,
                zy,
                s=55,
                c=z_colors,
                marker="*",
                edgecolors="#111111",
                linewidths=0.4,
                zorder=85,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="None",
                    marker="*",
                    color="w",
                    markerfacecolor="#888888",
                    markeredgecolor="#111111",
                    markersize=10,
                    label="Centroïde composante z (couleur = macro dominante)",
                )
            )

    if legend_handles:
        ax.legend(handles=legend_handles, loc="best", fontsize=7, frameon=True)


def plot_umap_plotly(
    coords: np.ndarray,
    lab: pd.DataFrame,
    *,
    label_col: str,
    title: str,
    out_html: PathLike,
    hover_label: str = "theme",
    max_legend: int = 40,
) -> "Any":
    """Scatter Plotly interactif (UMAP 2D)."""
    import plotly.express as px

    df = lab.copy()
    df["umap_x"] = coords[:, 0]
    df["umap_y"] = coords[:, 1]
    color_col = hover_label if hover_label in df.columns else label_col
    n_groups = df[color_col].nunique()
    fig = px.scatter(
        df,
        x="umap_x",
        y="umap_y",
        color=color_col,
        hover_data=[c for c in [label_col, "z_hat", "doc_id"] if c in df.columns],
        title=title,
        opacity=0.55,
        height=700,
        category_orders={color_col: sorted(df[color_col].astype(str).unique())},
    )
    if n_groups > max_legend:
        fig.update_layout(showlegend=False)
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(legend=dict(itemsizing="constant"))
    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return fig


def plot_embeddings_csv_pca_tsne(
    emb_csv: PathLike,
    meta_csv: PathLike,
    label_col: str = "pred_label",
    *,
    corpus_name: str = "BTP",
    save_fig: Optional[Callable[[str], Path]] = None,
    png_name: Optional[str] = None,
    max_points: int = 8000,
    seed: int = 42,
    show_macro_centroids: bool = True,
    pred_ok_col: str = "pred_ok",
    group_col: str = "accident_id",
) -> Optional[Path]:
    """PCA + t-SNE sur un CSV d'embeddings (dim_*) fusionné avec les métadonnées."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    from scgm_text.dataset_text_embeddings import load_filtered_metadata, merge_metadata_with_embeddings

    emb_path = Path(emb_csv)
    if not emb_path.is_file():
        return None

    meta_df = load_filtered_metadata(
        str(meta_csv),
        label_col=label_col,
        pred_ok_col=pred_ok_col,
        group_col=group_col,
    )
    merged, dim_cols = merge_metadata_with_embeddings(meta_df, str(emb_path), strict=False)
    sample_x = merged[dim_cols].to_numpy(dtype=np.float64)

    idx = sample_projection_indices(merged, label_col, max_points=max_points, seed=seed)
    sample_df = merged.loc[idx]
    sample_x = sample_x[idx]

    pca_xy = PCA(n_components=2, random_state=seed).fit_transform(sample_x)
    tsne_xy = TSNE(
        n_components=2,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    ).fit_transform(sample_x)

    if save_fig is None:
        def _show_only(name: str) -> Path:
            import matplotlib.pyplot as plt

            plt.tight_layout()
            plt.show()
            return Path(name)

        save_fig = _show_only

    out_name = png_name or "pca_tsne.png"
    return plot_projection_matplotlib(
        pca_xy,
        tsne_xy,
        sample_df,
        label_col,
        save_fig=save_fig,
        png_name=out_name,
        pca_title=f"PCA 2D — {corpus_name}",
        tsne_title=f"t-SNE 2D — {corpus_name}",
        show_macro_centroids=show_macro_centroids,
        show_z_centroids=False,
    )


def plot_tsne_per_macro_grid(
    X: np.ndarray,
    macro_labels: np.ndarray,
    *,
    corpus_name: str = "corpus",
    save_fig: Optional[Callable[[str], Path]] = None,
    png_name: str = "tsne_per_macro.png",
    seed: int = 42,
    max_points_per_macro: int = 2000,
    min_points: int = 10,
    macros_order: Sequence[str] = ("A0", "A1", "B", "C"),
    point_size: float = 5.0,
    figsize: Tuple[float, float] = (9.0, 8.0),
) -> Optional[Path]:
    """Grille 2×2 : t-SNE recalculé séparément sur chaque macro (structure intra-rôle)."""
    from sklearn.manifold import TSNE

    import matplotlib.pyplot as plt

    z = np.asarray(X, dtype=np.float64)
    labels = np.asarray(macro_labels).astype(str)
    if z.shape[0] != labels.shape[0]:
        raise ValueError("X et macro_labels doivent avoir la même longueur")

    macro_colors = _macro_color_map(macros_order)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes_flat = axes.ravel()
    rng = np.random.default_rng(seed)

    for ax, macro in zip(axes_flat, macros_order):
        mask = labels == macro
        n = int(mask.sum())
        if n < min_points:
            ax.set_title(f"t-SNE — {macro} (n={n}, insuffisant)")
            ax.axis("off")
            continue
        z_m = z[mask]
        if n > max_points_per_macro:
            pick = rng.choice(n, size=max_points_per_macro, replace=False)
            z_m = z_m[pick]
            n = max_points_per_macro
        tsne_xy = TSNE(
            n_components=2,
            random_state=seed,
            init="pca",
            learning_rate="auto",
            perplexity=min(30, n - 1),
        ).fit_transform(z_m)
        color = macro_colors.get(macro, "#888888")
        ax.scatter(tsne_xy[:, 0], tsne_xy[:, 1], s=point_size, alpha=0.5, c=color)
        ax.set_title(f"t-SNE — {macro} (n={n})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{corpus_name} — t-SNE par macro", y=0.995, fontsize=12)
    plt.tight_layout(pad=1.2)

    if save_fig is None:
        plt.show()
        return Path(png_name)

    return save_fig(png_name)


def plot_embeddings_csv_tsne_per_macro(
    emb_csv: PathLike,
    meta_csv: PathLike,
    label_col: str = "pred_label",
    *,
    corpus_name: str = "BTP",
    save_fig: Optional[Callable[[str], Path]] = None,
    png_name: Optional[str] = None,
    max_points: int = 8000,
    max_points_per_macro: int = 2000,
    min_points: int = 10,
    seed: int = 42,
    pred_ok_col: str = "pred_ok",
    group_col: str = "accident_id",
) -> Optional[Path]:
    """t-SNE 2D par macro (grille 2×2) depuis un CSV d'embeddings."""
    from scgm_text.dataset_text_embeddings import load_filtered_metadata, merge_metadata_with_embeddings

    emb_path = Path(emb_csv)
    if not emb_path.is_file():
        return None

    meta_df = load_filtered_metadata(
        str(meta_csv),
        label_col=label_col,
        pred_ok_col=pred_ok_col,
        group_col=group_col,
    )
    merged, dim_cols = merge_metadata_with_embeddings(meta_df, str(emb_path), strict=False)
    sample_x = merged[dim_cols].to_numpy(dtype=np.float64)

    idx = sample_projection_indices(merged, label_col, max_points=max_points, seed=seed)
    sample_df = merged.loc[idx]
    sample_x = sample_x[idx]
    macro_labels = sample_df[label_col].astype(str).to_numpy()

    if save_fig is None:

        def _show_only(name: str) -> Path:
            import matplotlib.pyplot as plt

            plt.tight_layout()
            plt.show()
            return Path(name)

        save_fig = _show_only

    return plot_tsne_per_macro_grid(
        sample_x,
        macro_labels,
        corpus_name=corpus_name,
        save_fig=save_fig,
        png_name=png_name or "tsne_per_macro.png",
        seed=seed,
        max_points_per_macro=max_points_per_macro,
        min_points=min_points,
    )


def resolve_softtriple_centers_csv(results_dir: PathLike) -> Optional[Path]:
    """Cherche softtriple_effective_centers.csv (run root ou checkpoint)."""
    root = Path(results_dir)
    for rel in (
        "centers/softtriple_effective_centers.csv",
        "checkpoints/best_model/centers/softtriple_effective_centers.csv",
    ):
        path = root / rel
        if path.is_file():
            return path
    return None


def load_softtriple_centers_matrix(centers_csv: PathLike) -> Tuple[np.ndarray, pd.DataFrame]:
    """Vecteurs dim_* et métadonnées (class_name, effective_center_id, …)."""
    df = pd.read_csv(centers_csv)
    dim_cols = sorted(
        [c for c in df.columns if c.startswith("dim_")],
        key=lambda name: int(name.split("_", 1)[1]),
    )
    if not dim_cols:
        raise ValueError(f"Aucune colonne dim_* dans {centers_csv}")
    vectors = df[dim_cols].to_numpy(dtype=np.float64)
    return vectors, df


def softtriple_centers_summary_table(centers_csv: PathLike) -> pd.DataFrame:
    """Tableau lisible : macro, id centre effectif, taille groupe, norme L2."""
    _, meta = load_softtriple_centers_matrix(centers_csv)
    dim_cols = sorted(
        [c for c in meta.columns if c.startswith("dim_")],
        key=lambda name: int(name.split("_", 1)[1]),
    )
    vectors = meta[dim_cols].to_numpy(dtype=np.float64)
    summary = meta[
        [c for c in ("class_id", "class_name", "effective_center_id", "group_size", "fold") if c in meta.columns]
    ].copy()
    summary["l2_norm"] = np.linalg.norm(vectors, axis=1)
    return summary.sort_values(
        [c for c in ("class_name", "effective_center_id") if c in summary.columns]
    ).reset_index(drop=True)


def overlay_softtriple_centers(
    ax,
    centers_xy: np.ndarray,
    centers_meta: pd.DataFrame,
    *,
    macros_order: Sequence[str] = ("A0", "A1", "B", "C"),
) -> None:
    """Superpose les centres SoftTriple (losange) sur PCA/t-SNE."""
    from matplotlib.lines import Line2D

    if centers_xy is None or len(centers_meta) == 0:
        return
    macro_colors = _macro_color_map(macros_order)
    xy = np.asarray(centers_xy, dtype=np.float64)
    for i in range(min(len(centers_meta), xy.shape[0])):
        row = centers_meta.iloc[i]
        macro = str(row.get("class_name", ""))
        color = macro_colors.get(macro, "#444444")
        ax.scatter(
            xy[i, 0],
            xy[i, 1],
            s=130,
            marker="D",
            c=[color],
            edgecolors="#111111",
            linewidths=1.0,
            zorder=95,
        )
        eff_id = row.get("effective_center_id", i)
        ax.annotate(
            f"{macro}#{int(eff_id)}",
            (xy[i, 0], xy[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            fontweight="bold",
            color="#111111",
            zorder=96,
        )
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker="D",
                color="w",
                markerfacecolor="#888888",
                markeredgecolor="#111111",
                markersize=8,
                label="Centre SoftTriple effectif",
            )
        ],
        loc="best",
        fontsize=7,
        frameon=True,
    )


def _project_softtriple_centers_2d(
    sample_x: np.ndarray,
    center_vectors: np.ndarray,
    *,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """PCA + t-SNE : embeddings + centres dans le même espace 2D."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    pca = PCA(n_components=2, random_state=seed)
    pca_xy = pca.fit_transform(sample_x)
    centers_pca = pca.transform(center_vectors)
    combined = np.vstack([sample_x, center_vectors])
    tsne_all = TSNE(
        n_components=2,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        perplexity=min(30, max(5, combined.shape[0] // 4)),
    ).fit_transform(combined)
    n_emb = sample_x.shape[0]
    return pca_xy, tsne_all[:n_emb], centers_pca, tsne_all[n_emb:]


def plot_embeddings_csv_pca_tsne_with_softtriple_centers(
    emb_csv: PathLike,
    meta_csv: PathLike,
    label_col: str = "pred_label",
    *,
    results_dir: PathLike,
    corpus_name: str = "BTP",
    save_fig: Optional[Callable[[str], Path]] = None,
    png_name: Optional[str] = None,
    max_points: int = 8000,
    seed: int = 42,
    show_macro_centroids: bool = True,
    pred_ok_col: str = "pred_ok",
    group_col: str = "accident_id",
) -> Optional[Path]:
    """PCA/t-SNE embeddings + centres SoftTriple effectifs superposés."""
    from scgm_text.dataset_text_embeddings import load_filtered_metadata, merge_metadata_with_embeddings

    emb_path = Path(emb_csv)
    centers_csv = resolve_softtriple_centers_csv(results_dir)
    if not emb_path.is_file() or centers_csv is None:
        return None

    meta_df = load_filtered_metadata(
        str(meta_csv),
        label_col=label_col,
        pred_ok_col=pred_ok_col,
        group_col=group_col,
    )
    merged, dim_cols = merge_metadata_with_embeddings(meta_df, str(emb_path), strict=False)
    center_vectors, centers_meta = load_softtriple_centers_matrix(centers_csv)
    sample_x = merged[dim_cols].to_numpy(dtype=np.float64)

    idx = sample_projection_indices(merged, label_col, max_points=max_points, seed=seed)
    sample_df = merged.loc[idx]
    sample_x = sample_x[idx]

    pca_xy, tsne_xy, centers_pca, centers_tsne = _project_softtriple_centers_2d(
        sample_x, center_vectors, seed=seed
    )

    if save_fig is None:

        def _show_only(name: str) -> Path:
            import matplotlib.pyplot as plt

            plt.tight_layout()
            plt.show()
            return Path(name)

        save_fig = _show_only

    out_name = png_name or "pca_tsne_centers.png"
    return plot_projection_matplotlib(
        pca_xy,
        tsne_xy,
        sample_df,
        label_col,
        save_fig=save_fig,
        png_name=out_name,
        pca_title=f"PCA 2D — {corpus_name}",
        tsne_title=f"t-SNE 2D — {corpus_name}",
        show_macro_centroids=show_macro_centroids,
        show_z_centroids=False,
        softtriple_centers_pca=centers_pca,
        softtriple_centers_tsne=centers_tsne,
        softtriple_centers_meta=centers_meta,
    )


def plot_projection_matplotlib(
    pca_xy: np.ndarray,
    tsne_xy: np.ndarray,
    sample_df: pd.DataFrame,
    label_col: str,
    *,
    save_fig: Callable[[str], Path],
    png_name: str = "05_projection_macro.png",
    pca_title: str = "PCA 2D (macro)",
    tsne_title: str = "t-SNE 2D (macro)",
    show_macro_centroids: bool = True,
    show_z_centroids: bool = False,
    z_col: str = "z_hat",
    themes_z: Optional[pd.DataFrame] = None,
    softtriple_centers_pca: Optional[np.ndarray] = None,
    softtriple_centers_tsne: Optional[np.ndarray] = None,
    softtriple_centers_meta: Optional[pd.DataFrame] = None,
    point_size: float = 4.0,
    figsize: Tuple[float, float] = (11.0, 4.0),
) -> Path:
    """PCA + t-SNE côte à côte (matplotlib statique)."""
    import matplotlib.pyplot as plt

    macros_order = ["A0", "A1", "B", "C"]
    macro_colors = _macro_color_map(macros_order)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    sample_reset = sample_df.reset_index(drop=True)
    for label in macros_order:
        color = macro_colors[label]
        mask = sample_reset[label_col].values == label
        axes[0].scatter(
            pca_xy[mask, 0], pca_xy[mask, 1], s=point_size, alpha=0.45, label=label, c=[color]
        )
        axes[1].scatter(
            tsne_xy[mask, 0], tsne_xy[mask, 1], s=point_size, alpha=0.45, label=label, c=[color]
        )
    axes[0].set_title(pca_title, fontsize=11)
    axes[1].set_title(tsne_title, fontsize=11)
    if show_macro_centroids or show_z_centroids:
        overlay_projection_centroids(
            axes[0],
            pca_xy,
            sample_reset,
            label_col,
            z_col=z_col,
            themes_z=themes_z,
            show_macro=show_macro_centroids,
            show_z=show_z_centroids,
        )
        overlay_projection_centroids(
            axes[1],
            tsne_xy,
            sample_reset,
            label_col,
            z_col=z_col,
            themes_z=themes_z,
            show_macro=show_macro_centroids,
            show_z=show_z_centroids,
        )
    if softtriple_centers_pca is not None and softtriple_centers_meta is not None:
        overlay_softtriple_centers(axes[0], softtriple_centers_pca, softtriple_centers_meta)
    if softtriple_centers_tsne is not None and softtriple_centers_meta is not None:
        overlay_softtriple_centers(axes[1], softtriple_centers_tsne, softtriple_centers_meta)
    if not (show_macro_centroids or show_z_centroids or softtriple_centers_pca is not None):
        axes[0].legend(fontsize=8, loc="best", markerscale=2)
        axes[1].legend(fontsize=8, loc="best", markerscale=2)
    fig.tight_layout(pad=1.2)
    return save_fig(png_name)


def plot_projection_plotly(
    pca_xy: np.ndarray,
    tsne_xy: np.ndarray,
    sample_df: pd.DataFrame,
    label_col: str,
    *,
    figures_dir: PathLike,
) -> Tuple[Any, Any]:
    """PCA et t-SNE en figures Plotly séparées (HTML)."""
    import plotly.express as px

    figures_dir = Path(figures_dir)
    pal = _macro_palette()
    base = sample_df.copy().reset_index(drop=True)
    base[label_col] = base[label_col].astype(str)

    def _one(xy: np.ndarray, method: str, fname: str):
        d = base.copy()
        d["x"] = xy[:, 0]
        d["y"] = xy[:, 1]
        fig = px.scatter(
            d,
            x="x",
            y="y",
            color=label_col,
            color_discrete_map=pal,
            hover_data=[c for c in ["z_hat", "doc_id"] if c in d.columns],
            title=f"{method} 2D — segments (macro {label_col})",
            opacity=0.55,
            height=480,
        )
        fig.update_traces(marker=dict(size=4))
        path = figures_dir / fname
        fig.write_html(str(path), include_plotlyjs="cdn")
        return fig, path

    return _one(pca_xy, "PCA", "05_projection_pca_interactive.html"), _one(
        tsne_xy, "t-SNE", "05_projection_tsne_interactive.html"
    )


def plot_training_geometry_curves(
    logs: pd.DataFrame,
    *,
    save_fig: Callable[[str], Path],
) -> None:
    """Courbes d'entraînement : losses, validation, eta² macro."""
    import matplotlib.pyplot as plt

    eta_cols = [
        c
        for c in (
            "val_eta2_macro_balanced",
            "val_eta2_weighted",
            "val_eta2_macro_balanced_perc",
            "train_eta2_macro_balanced",
        )
        if c in logs.columns
    ]
    if not eta_cols and "train_loss" not in logs.columns:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    loss_cols = [c for c in ["train_loss", "loss_macro", "loss_latent"] if c in logs.columns]
    if loss_cols:
        logs.plot(x="epoch", y=loss_cols, ax=axes[0, 0], marker="o", markersize=3)
        axes[0, 0].set_title("Pertes d'entraînement")
    val_cols = [c for c in ["val_acc", "val_macro_f1", "val_balanced_acc"] if c in logs.columns]
    if val_cols:
        logs.plot(x="epoch", y=val_cols, ax=axes[0, 1], marker="o", markersize=3)
        axes[0, 1].set_title("Validation macro")
    if eta_cols:
        logs.plot(x="epoch", y=eta_cols, ax=axes[1, 0], marker="o", markersize=3)
        axes[1, 0].set_title("Eta² macro (val/train)")
        if len(eta_cols) == 1:
            col = eta_cols[0]
            logs.plot(x="epoch", y=col, ax=axes[1, 1], marker="s", markersize=3, color="#27ae60")
            axes[1, 1].set_title(col)
        else:
            for col, ax, color in zip(
                eta_cols[:3],
                [axes[1, 1]] * min(3, len(eta_cols)),
                ["#27ae60", "#3498db", "#9b59b6"],
            ):
                logs.plot(x="epoch", y=col, ax=ax, marker="s", markersize=3, label=col, color=color)
            axes[1, 1].legend(fontsize=8)
            axes[1, 1].set_title("Eta² — courbes séparées")
    save_fig("04b_training_geometry.png")


def plot_evaluation_geometry_dashboard(
    metrics_table: pd.DataFrame,
    *,
    figures_dir: PathLike,
    save_fig: Callable[[str], Path],
) -> List[Path]:
    """Dashboard eta2 macro-balanced / weighted et inertie intra-macro."""
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
    import seaborn as sns

    figures_dir = Path(figures_dir)
    saved: List[Path] = []
    if metrics_table.empty:
        return saved

    eta_cols = ("eta2_macro_balanced", "eta2_weighted")
    missing_eta = [c for c in eta_cols if c not in metrics_table.columns]
    if missing_eta:
        if "eta2_macro_balanced_perc" not in metrics_table.columns and "eta2_macro_balanced" not in metrics_table.columns:
            raise KeyError(
                "Colonnes eta2_macro_balanced_perc / eta2_macro_balanced absentes. "
                "Relancez l'évaluation ou le post-traitement contrastif."
            )
        raise KeyError(
            f"Colonnes manquantes dans metrics_table : {missing_eta}. "
            "Relancez l'évaluation (evaluate_scgm_text.py) avant les graphiques."
        )

    macro_names = ["A0", "A1", "B", "C"]
    w_cols = [f"W_{m}" for m in macro_names]
    n_cols = [f"n_{m}" for m in macro_names]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    methods = (
        metrics_table["method"].astype(str).tolist()
        if "method" in metrics_table.columns
        else [str(i) for i in range(len(metrics_table))]
    )

    x = np.arange(len(methods))
    width = 0.35
    axes[0, 0].bar(
        x - width / 2,
        metrics_table["eta2_macro_balanced"].astype(float),
        width,
        label="eta2_macro_balanced",
        color="#3498db",
    )
    axes[0, 0].bar(
        x + width / 2,
        metrics_table["eta2_weighted"].astype(float),
        width,
        label="eta2_weighted",
        color="#e74c3c",
    )
    axes[0, 0].set_xticks(x, methods, rotation=45, ha="right")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_title("Eta² structuration macro (0–1)")
    axes[0, 0].legend(fontsize=8)

    row0 = metrics_table.iloc[0]
    w_vals = [float(row0.get(c, float("nan"))) for c in w_cols]
    n_vals = [int(row0.get(c, 0)) for c in n_cols]
    axes[0, 1].bar(macro_names, w_vals, color=sns.color_palette("Set2", 4))
    axes[0, 1].set_title(f"Inertie intra W(c) — {row0.get('method', '')}")
    axes[0, 1].set_ylabel("||z - mu_c||²")

    axes[1, 0].bar(macro_names, n_vals, color=sns.color_palette("Pastel1", 4))
    axes[1, 0].set_title("Effectifs par macro")
    axes[1, 0].set_ylabel("n")

    heat_data = metrics_table[w_cols].astype(float)
    if "method" in metrics_table.columns:
        heat_data.index = metrics_table["method"].astype(str)
    sns.heatmap(heat_data, annot=True, fmt=".3f", cmap="YlOrRd", ax=axes[1, 1])
    axes[1, 1].set_title("Heatmap inertie intra W(c)")
    saved.append(save_fig("09_eta2_geometry_dashboard.png"))

    if "method" in metrics_table.columns:
        eta_long = metrics_table.melt(
            id_vars=["method"],
            value_vars=["eta2_macro_balanced", "eta2_weighted"],
            var_name="metric",
            value_name="value",
        )
        fig_eta = px.bar(
            eta_long,
            x="method",
            y="value",
            color="metric",
            barmode="group",
            title="Eta² structuration macro (0–1)",
            height=500,
        )
    else:
        eta_long = pd.DataFrame(
            {
                "metric": ["eta2_macro_balanced", "eta2_weighted"],
                "value": metrics_table[["eta2_macro_balanced", "eta2_weighted"]]
                .iloc[0]
                .astype(float)
                .tolist(),
            }
        )
        fig_eta = px.bar(eta_long, x="metric", y="value", title="Eta² structuration macro (0–1)", height=500)
    fig_eta.update_layout(yaxis=dict(range=[0, 1.05]))
    p_eta = figures_dir / "09_eta2_macro_interactive.html"
    fig_eta.write_html(str(p_eta), include_plotlyjs="cdn")
    saved.append(p_eta)

    return saved


def plot_embedding_umap_by_macro(
    embeddings: np.ndarray,
    meta: pd.DataFrame,
    label_col: str,
    *,
    figures_dir: PathLike,
    save_fig: Callable[[str], Path],
    max_points: int = 12000,
    seed: int = 42,
    title: str = "UMAP — embedding brut (couleur = macro)",
    png_name: str = "09_raw_embedding_umap.png",
    html_name: str = "09_raw_embedding_umap_interactive.html",
    show_macro_centroids: bool = True,
    include_plotly: bool = True,
) -> List[Path]:
    """UMAP 2D d'un nuage d'embeddings, coloré par macro (matplotlib + Plotly optionnel)."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from umap import UMAP

    figures_dir = Path(figures_dir)
    saved: List[Path] = []
    x = np.asarray(embeddings, dtype=np.float64)
    if len(meta) != x.shape[0]:
        raise ValueError(f"meta ({len(meta)}) and embeddings ({x.shape[0]}) length mismatch.")

    idx = sample_projection_indices(meta, label_col, max_points=max_points, seed=seed)
    sample_df = meta.loc[idx].copy()
    sample_x = x[idx]
    macro_labels = sample_df[label_col].astype(str).to_numpy()

    reducer = UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(sample_x)

    fig, ax = plt.subplots(figsize=(10, 8))
    palette = dict(zip(["A0", "A1", "B", "C"], sns.color_palette("Set2", 4)))
    for macro in ["A0", "A1", "B", "C"]:
        mask = macro_labels == macro
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=6,
            alpha=0.45,
            label=macro,
            c=[palette[macro]],
        )
    if show_macro_centroids:
        cx, cy, names, macro_to_color = macro_umap_centroids(coords, macro_labels)
        if names:
            ax.scatter(
                cx,
                cy,
                s=200,
                c=[macro_to_color[m] for m in names],
                marker="P",
                edgecolors="#111111",
                linewidths=1.0,
                zorder=10,
            )
            for xi, yi, m in zip(cx, cy, names):
                ax.annotate(
                    m,
                    (xi, yi),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold",
                    zorder=11,
                )
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(title=label_col, markerscale=2)
    saved.append(save_fig(png_name))

    if include_plotly:
        html_path = figures_dir / html_name
        plot_umap_plotly(
            coords,
            sample_df,
            label_col=label_col,
            title=title,
            out_html=html_path,
            hover_label=label_col,
            max_legend=10,
        )
        saved.append(html_path)
    return saved


def display_plotly_html(html_path: PathLike) -> None:
    """Affiche un fichier HTML Plotly dans Jupyter."""
    from IPython.display import HTML, display

    path = Path(html_path)
    if path.is_file():
        display(HTML(path.read_text(encoding="utf-8")))


_KFOLD_BAR_METRICS: Tuple[str, ...] = (
    "eta2_macro_balanced_perc",
    "eta2_macro_balanced",
)


def _resolve_kfold_per_fold_metric_cols(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for key in _KFOLD_BAR_METRICS:
        if key in df.columns:
            cols.append(key)
        elif key == "eta2_macro_balanced" and "val_eta2_macro_balanced" in df.columns:
            cols.append("val_eta2_macro_balanced")
    return cols


def plot_kfold_metrics_bars(
    kfold_per_fold: pd.DataFrame,
    *,
    save_fig: Callable[[str], Path],
) -> Optional[Path]:
    """Barres groupées des métriques géométriques par fold (validation)."""
    import matplotlib.pyplot as plt

    if kfold_per_fold.empty or "fold_id" not in kfold_per_fold.columns:
        return None
    metric_cols = _resolve_kfold_per_fold_metric_cols(kfold_per_fold)
    if not metric_cols:
        return None

    df = kfold_per_fold.sort_values("fold_id")
    folds = df["fold_id"].astype(int).tolist()
    x = np.arange(len(folds))
    width = 0.8 / len(metric_cols)
    fig, ax = plt.subplots(figsize=(max(8, 2 * len(folds)), 5))
    for i, col in enumerate(metric_cols):
        offset = (i - (len(metric_cols) - 1) / 2) * width
        vals = pd.to_numeric(df[col], errors="coerce").astype(float)
        ax.bar(x + offset, vals, width=width, label=col.replace("val_", ""))
    ax.set_xticks(x, [f"fold {f}" for f in folds])
    ax.set_title("K-fold — métriques validation par fold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return save_fig("kfold_metrics_by_fold.png")


def plot_kfold_val_curves(
    output_path: PathLike,
    *,
    save_fig: Callable[[str], Path],
    folds_subdir: str = "folds",
    log_name: str = "train_log.csv",
) -> Optional[Path]:
    """Courbes val_eta2_macro_balanced_perc (et η²) vs epoch, une ligne par fold."""
    import matplotlib.pyplot as plt

    root = Path(output_path)
    folds_dir = root / folds_subdir
    if not folds_dir.is_dir():
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    has_any = False
    for fold_dir in sorted(folds_dir.glob("fold_*")):
        log_path = fold_dir / "metrics" / log_name
        if not log_path.is_file():
            continue
        log = pd.read_csv(log_path)
        if "epoch" not in log.columns:
            continue
        fold_id = fold_dir.name.replace("fold_", "")
        has_any = True
        if "val_eta2_macro_balanced_perc" in log.columns:
            axes[0].plot(
                log["epoch"],
                log["val_eta2_macro_balanced_perc"],
                marker="o",
                markersize=3,
                label=f"fold {fold_id}",
            )
        eta_col = "val_eta2_macro_balanced" if "val_eta2_macro_balanced" in log.columns else None
        if eta_col:
            axes[1].plot(
                log["epoch"],
                log[eta_col],
                marker="o",
                markersize=3,
                label=f"fold {fold_id}",
            )
    if not has_any:
        plt.close(fig)
        return None
    axes[0].set_title("δ_macro validation (%)")
    axes[0].set_xlabel("epoch")
    axes[0].legend(fontsize=8)
    axes[1].set_title("η² macro balanced (validation)")
    axes[1].set_xlabel("epoch")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    return save_fig("kfold_val_curves.png")


def plot_kfold_summary_errorbars(
    kfold_summary: pd.DataFrame,
    *,
    save_fig: Callable[[str], Path],
) -> Optional[Path]:
    """Barres μ±σ depuis kfold_summary.csv."""
    import matplotlib.pyplot as plt

    if kfold_summary.empty:
        return None
    row = kfold_summary.iloc[0]
    pairs: List[Tuple[str, str, str]] = []
    for key in _KFOLD_BAR_METRICS:
        mean_col = f"mean_{key}"
        std_col = f"std_{key}"
        if mean_col in row.index:
            pairs.append((key, mean_col, std_col))
        elif key == "eta2_macro_balanced" and "mean_val_eta2_macro_balanced" in row.index:
            pairs.append(
                ("val_eta2_macro_balanced", "mean_val_eta2_macro_balanced", "std_val_eta2_macro_balanced")
            )
    if not pairs:
        return None

    labels = [p[0].replace("val_", "") for p in pairs]
    means = [float(row[p[1]]) for p in pairs]
    stds = [float(row.get(p[2], 0.0)) for p in pairs]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 4))
    ax.bar(x, means, yerr=stds, capsize=4, color="#3498db", alpha=0.85)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_title("K-fold validation — μ ± σ")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return save_fig("kfold_summary_errorbars.png")


def load_projected_embeddings_pair(
    npy_path: PathLike,
    meta_path: PathLike,
) -> Optional[tuple[np.ndarray, pd.DataFrame]]:
    """Charge ``projected_*.npy`` + CSV métadonnées (représentations avant classif. LR)."""
    npy_p = Path(npy_path)
    meta_p = Path(meta_path)
    if not npy_p.is_file() or not meta_p.is_file():
        return None
    projected = np.load(npy_p)
    meta = pd.read_csv(meta_p)
    if len(meta) != projected.shape[0]:
        print(f"Attention : meta ({len(meta)}) vs projected ({projected.shape[0]})")
    return projected, meta


def plot_projected_embeddings_pca_tsne(
    npy_path: PathLike,
    meta_path: PathLike,
    label_col: str,
    *,
    corpus_name: str,
    save_fig: Callable[[str], Path],
    figures_dir: PathLike,
    max_points: int = 8000,
    seed: int = 42,
    png_name: str = "projection_pca_tsne.png",
    show_macro_centroids: bool = False,
    show_z_centroids: bool = False,
    z_col: str = "z_hat",
    themes_z: Optional[pd.DataFrame] = None,
    point_size: float = 4.0,
    figsize: Tuple[float, float] = (11.0, 4.0),
    include_plotly: bool = True,
) -> Optional[List[Path]]:
    """PCA + t-SNE 2D sur embeddings projetés (avant régression logistique sklearn)."""
    pair = load_projected_embeddings_pair(npy_path, meta_path)
    if pair is None:
        return None
    projected, meta = pair
    if label_col not in meta.columns:
        print(f"Colonne {label_col} absente de {meta_path}")
        return None
    return plot_corpus_projections(
        projected,
        meta,
        label_col,
        corpus_name=corpus_name,
        save_fig=save_fig,
        figures_dir=figures_dir,
        max_points=max_points,
        seed=seed,
        png_name=png_name,
        show_macro_centroids=show_macro_centroids,
        show_z_centroids=show_z_centroids,
        z_col=z_col,
        themes_z=themes_z,
        point_size=point_size,
        figsize=figsize,
        include_plotly=include_plotly,
    )


def plot_corpus_projections(
    projected: np.ndarray,
    meta: pd.DataFrame,
    label_col: str,
    *,
    corpus_name: str = "Test métallurgie",
    save_fig: Callable[[str], Path],
    figures_dir: PathLike,
    max_points: int = 8000,
    seed: int = 42,
    png_name: str = "10_test_projection_macro.png",
    show_macro_centroids: bool = True,
    show_z_centroids: bool = True,
    z_col: str = "z_hat",
    themes_z: Optional[pd.DataFrame] = None,
    point_size: float = 4.0,
    figsize: Tuple[float, float] = (11.0, 4.0),
    include_plotly: bool = True,
) -> List[Path]:
    """PCA + t-SNE 2D sur embeddings SCGM projetés (matplotlib + Plotly optionnel)."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    figures_dir = Path(figures_dir)
    x = np.asarray(projected, dtype=np.float64)
    if len(meta) != x.shape[0]:
        raise ValueError(f"meta ({len(meta)}) vs projected ({x.shape[0]})")

    idx = sample_projection_indices(meta, label_col, max_points=max_points, seed=seed)
    sample_df = meta.loc[idx].copy().reset_index(drop=True)
    sample_x = x[idx]

    pca_xy = PCA(n_components=2, random_state=seed).fit_transform(sample_x)
    tsne_xy = TSNE(n_components=2, random_state=seed, perplexity=min(30, len(sample_x) - 1)).fit_transform(
        sample_x
    )

    saved: List[Path] = []
    saved.append(
        plot_projection_matplotlib(
            pca_xy,
            tsne_xy,
            sample_df,
            label_col,
            save_fig=save_fig,
            png_name=png_name,
            pca_title=f"PCA 2D — {corpus_name}",
            tsne_title=f"t-SNE 2D — {corpus_name}",
            show_macro_centroids=show_macro_centroids,
            show_z_centroids=show_z_centroids,
            z_col=z_col,
            themes_z=themes_z,
            point_size=point_size,
            figsize=figsize,
        )
    )
    if include_plotly:
        pca_pair, tsne_pair = plot_projection_plotly(
            pca_xy, tsne_xy, sample_df, label_col, figures_dir=figures_dir
        )
        saved.extend([pca_pair[1], tsne_pair[1]])
    return saved


def plot_corpus_umap(
    projected: np.ndarray,
    meta: pd.DataFrame,
    label_col: str,
    *,
    corpus_name: str = "Test métallurgie",
    save_fig: Callable[[str], Path],
    figures_dir: PathLike,
    max_points: int = 12000,
    seed: int = 42,
    png_name: str = "10_test_umap.png",
    html_name: str = "10_test_umap_interactive.html",
) -> List[Path]:
    """UMAP sur corpus projeté SCGM."""
    return plot_embedding_umap_by_macro(
        projected,
        meta,
        label_col,
        figures_dir=figures_dir,
        save_fig=save_fig,
        max_points=max_points,
        seed=seed,
        title=f"UMAP — {corpus_name} (SCGM projeté, couleur = macro)",
        png_name=png_name,
        html_name=html_name,
    )


def plot_btp_test_umap_pair(
    btp_projected: np.ndarray,
    btp_meta: pd.DataFrame,
    test_projected: np.ndarray,
    test_meta: pd.DataFrame,
    label_col: str,
    *,
    save_fig: Callable[[str], Path],
    figures_dir: PathLike,
    max_points: int = 8000,
    seed: int = 42,
) -> Optional[Path]:
    """Figure 2×2 UMAP BTP vs test (matplotlib)."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from umap import UMAP

    figures_dir = Path(figures_dir)
    palette = dict(zip(["A0", "A1", "B", "C"], sns.color_palette("Set2", 4)))

    def _umap_panel(ax, emb: np.ndarray, meta: pd.DataFrame, title: str) -> None:
        idx = sample_projection_indices(meta, label_col, max_points=max_points, seed=seed)
        sample_df = meta.loc[idx]
        sample_x = emb[idx]
        coords = UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1).fit_transform(sample_x)
        macros = sample_df[label_col].astype(str).to_numpy()
        for macro in ["A0", "A1", "B", "C"]:
            mask = macros == macro
            if np.any(mask):
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=5,
                    alpha=0.4,
                    label=macro,
                    c=[palette[macro]],
                )
        ax.set_title(title)
        ax.legend(fontsize=7, markerscale=2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _umap_panel(axes[0, 0], btp_projected, btp_meta, "BTP — UMAP")
    _umap_panel(axes[0, 1], test_projected, test_meta, "Test métallurgie — UMAP")
    axes[1, 0].axis("off")
    axes[1, 1].axis("off")
    plt.tight_layout()
    return save_fig("10_btp_test_umap_pair.png")


def plot_topics_distribution_by_macro(
    themes_z: pd.DataFrame,
    *,
    save_fig: Optional[Callable[[str], Path]] = None,
    png_name: str = "topics_by_macro.png",
) -> None:
    """Barplot : somme ``n_units`` par ``dominant_macro``."""
    import matplotlib.pyplot as plt

    if "dominant_macro" not in themes_z.columns or "n_units" not in themes_z.columns:
        print("themes_by_z : colonnes dominant_macro ou n_units absentes")
        return

    agg = (
        themes_z[themes_z["dominant_macro"].astype(str).str.len() > 0]
        .groupby("dominant_macro", as_index=False)["n_units"]
        .sum()
        .sort_values("dominant_macro")
    )
    if agg.empty:
        print("Aucune macro dominante à tracer")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(agg["dominant_macro"].astype(str), agg["n_units"].astype(float))
    ax.set_xlabel("Macro dominante")
    ax.set_ylabel("Segments (n_units)")
    ax.set_title("Distribution des topics par macro")
    plt.tight_layout()
    if save_fig is not None:
        save_fig(png_name)
    else:
        plt.show()


def macro_counts_per_z(
    meta_df: pd.DataFrame,
    *,
    z_col: str = "z_hat",
    label_col: str = "pred_label",
    macros: Sequence[str] = ("A0", "A1", "B", "C"),
) -> pd.DataFrame:
    """Effectifs par composante z et macro (colonnes A0..C + n_total)."""
    missing = [c for c in (z_col, label_col) if c not in meta_df.columns]
    if missing:
        raise ValueError(f"macro_counts_per_z : colonnes manquantes {missing}")
    ct = pd.crosstab(meta_df[z_col].astype(int), meta_df[label_col].astype(str))
    for m in macros:
        if m not in ct.columns:
            ct[m] = 0
    ct = ct.reindex(columns=[m for m in macros], fill_value=0).fillna(0).astype(int)
    ct["n_total"] = ct[list(macros)].sum(axis=1).astype(int)
    ct = ct.reset_index()
    first = str(ct.columns[0])
    if first != "z_id":
        ct = ct.rename(columns={first: "z_id"})
    return ct


_N_MACRO_COLS = {"A0": "n_A0", "A1": "n_A1", "B": "n_B", "C": "n_C"}


def plot_topics_n_units_by_z(
    themes_z: pd.DataFrame,
    *,
    metadata_df: Optional[pd.DataFrame] = None,
    z_col: str = "z_hat",
    label_col: str = "pred_label",
    save_fig: Optional[Callable[[str], Path]] = None,
    png_name: str = "topics_n_units_by_z.png",
) -> None:
    """Barres empilées : ``n_units`` par ``z_id`` et répartition A0–C (macro ``pred_label``)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if "z_id" not in themes_z.columns or "n_units" not in themes_z.columns:
        print("themes_by_z : colonnes z_id ou n_units absentes")
        return

    df = themes_z.sort_values("z_id").copy()
    macros_order = ("A0", "A1", "B", "C")
    n_cols = [_N_MACRO_COLS[m] for m in macros_order]
    macro_to_color = _macro_color_map(macros_order)

    use_stacked = all(c in df.columns for c in n_cols)
    if not use_stacked and metadata_df is not None:
        if z_col in metadata_df.columns and label_col in metadata_df.columns:
            try:
                counts = macro_counts_per_z(metadata_df, z_col=z_col, label_col=label_col)
                rename = {m: _N_MACRO_COLS[m] for m in macros_order}
                counts = counts.rename(columns=rename)
                keep = ["z_id"] + n_cols
                df = df.drop(columns=[c for c in n_cols if c in df.columns], errors="ignore")
                df = df.merge(counts[keep], on="z_id", how="left")
                df[n_cols] = df[n_cols].fillna(0).astype(int)
                use_stacked = True
            except ValueError as exc:
                print(f"(avertissement) répartition macros par z : {exc}")
        else:
            print(
                f"(info) metadata sans {z_col!r} / {label_col!r} — repli barres par macro dominante"
            )

    fig_w = max(10.0, len(df) * 0.28)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    if use_stacked:
        x = np.arange(len(df))
        parts = df[n_cols].astype(float).to_numpy().T
        totals_meta = parts.sum(axis=0)
        n_units_arr = df["n_units"].astype(float).to_numpy()
        scale = np.ones(len(df))
        ok = totals_meta > 1e-9
        scale[ok] = n_units_arr[ok] / totals_meta[ok]
        parts = parts * scale
        bottom = np.zeros(len(df))
        for i, m in enumerate(macros_order):
            heights = parts[i]
            ax.bar(
                x,
                heights,
                bottom=bottom,
                label=m,
                color=macro_to_color[m],
                edgecolor="#111111",
                linewidth=0.35,
            )
            bottom = bottom + heights
        ax.set_xticks(x, df["z_id"].astype(str), rotation=90, fontsize=7)
        ymax = float(bottom.max()) if len(bottom) else 0.0
        pad = max(ymax * 0.02, 1.0)
        for xi, total in enumerate(df["n_units"].astype(int).to_numpy()):
            if total > 0:
                ax.text(xi, float(bottom[xi]) + pad * 0.25, str(int(total)), ha="center", va="bottom", fontsize=6)
        ax.set_title("Effectifs par composante z — répartition des macros")
        legend_title = "Macro (pred_label)"
    else:
        print(
            "(info) Barres par macro dominante : régénérer themes_by_z (export_scgm_text_outputs.py, legacy) "
            "ou fournir metadata (z_hat, pred_label)."
        )
        if "dominant_macro" in df.columns:
            bar_colors = [
                macro_to_color.get(str(m).strip(), "#888888") for m in df["dominant_macro"].astype(str)
            ]
        else:
            bar_colors = ["#888888"] * len(df)
        ax.bar(df["z_id"].astype(str), df["n_units"].astype(float), color=bar_colors)
        plt.xticks(rotation=90, fontsize=7)
        ax.set_title("Effectif par composante z (couleur = macro dominante)")
        legend_title = "Macro dominante"

    ax.set_xlabel("Composante z")
    ax.set_ylabel("Segments (n_units)")
    legend_patches = [
        Patch(facecolor=macro_to_color[m], edgecolor="#111111", label=m) for m in macros_order
    ]
    ax.legend(handles=legend_patches, title=legend_title, loc="upper right", fontsize=8)
    plt.tight_layout()
    if save_fig is not None:
        save_fig(png_name)
    else:
        plt.show()
