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
            label=f"{m} — centroïde macro (moyenne UMAP)",
        )
        for m in names
    ]
    ax.legend(handles=leg_handles, loc="lower left", frameon=True, title="Macro (pred_label)", fontsize=8, title_fontsize=9)
    return fig, ax


def macro_umap_centroids(
    coords: np.ndarray,
    macro_labels: np.ndarray,
    macros_order: Sequence[str] = ("A0", "A1", "B", "C"),
) -> Tuple[List[float], List[float], List[str], Dict[str, str]]:
    """Centroïdes UMAP par macro pour overlay matplotlib."""
    from matplotlib import colors as mcolors
    import seaborn as sns

    pal_hex = [mcolors.to_hex(c) for c in sns.color_palette("Set2", 4)]
    macro_to_color = dict(zip(macros_order, pal_hex))
    cx, cy, names = [], [], []
    for m in macros_order:
        mask = macro_labels == m
        if not np.any(mask):
            continue
        mu = coords[mask].mean(axis=0)
        cx.append(float(mu[0]))
        cy.append(float(mu[1]))
        names.append(m)
    return cx, cy, names, macro_to_color


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
) -> Path:
    """PCA + t-SNE côte à côte (matplotlib statique)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, color in zip(["A0", "A1", "B", "C"], sns.color_palette("Set2", 4)):
        mask = sample_df[label_col].values == label
        axes[0].scatter(pca_xy[mask, 0], pca_xy[mask, 1], s=8, alpha=0.5, label=label, c=[color])
        axes[1].scatter(tsne_xy[mask, 0], tsne_xy[mask, 1], s=8, alpha=0.5, label=label, c=[color])
    axes[0].set_title(pca_title)
    axes[1].set_title(tsne_title)
    axes[0].legend()
    axes[1].legend()
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
            opacity=0.6,
            height=650,
        )
        fig.update_traces(marker=dict(size=6))
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
    """Courbes d'entraînement : losses, validation, géométrie (RankMe, C1, C10)."""
    import matplotlib.pyplot as plt

    geom_cols = [c for c in ["rankme_global", "c1_global", "c10_global"] if c in logs.columns]
    if not geom_cols:
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
    logs.plot(x="epoch", y=geom_cols, ax=axes[1, 0], marker="o", markersize=3)
    axes[1, 0].set_title("Géométrie globale (RankMe, C1, C10)")
    for col, ax, color in zip(geom_cols, [axes[1, 1]] * len(geom_cols), ["#2ecc71", "#3498db", "#9b59b6"]):
        logs.plot(x="epoch", y=col, ax=ax, marker="s", markersize=3, label=col, color=color)
    axes[1, 1].set_title("Géométrie — courbes séparées")
    axes[1, 1].legend(fontsize=8)
    save_fig("04b_training_geometry.png")

    fig2, axes2 = plt.subplots(1, len(geom_cols), figsize=(4 * len(geom_cols), 3.5))
    if len(geom_cols) == 1:
        axes2 = [axes2]
    for ax, col in zip(axes2, geom_cols):
        logs.plot(x="epoch", y=col, ax=ax, marker="o", color="#27ae60")
        ax.set_title(col)
    save_fig("04c_rankme_c1_c10_epochs.png")


def plot_evaluation_geometry_dashboard(
    metrics_table: pd.DataFrame,
    *,
    figures_dir: PathLike,
    save_fig: Callable[[str], Path],
) -> List[Path]:
    """Dashboard eta2 macro-balanced / weighted, inertie intra-macro et RankMe / C1 / C10."""
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
        if "delta_macro_pct" not in metrics_table.columns and "eta2_macro_balanced" not in metrics_table.columns:
            raise KeyError(
                "Colonnes delta_macro_pct / eta2_macro_balanced absentes. "
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

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, title in zip(
        axes2,
        ["rankme_global", "c1_global", "c10_global"],
        ["RankMe effectif", "C1 (énergie PCA)", "C10 (énergie PCA)"],
    ):
        if col in metrics_table.columns:
            if "method" in metrics_table.columns and len(metrics_table) > 1:
                metrics_table.plot(x="method", y=col, kind="bar", ax=ax, legend=False, rot=45)
            else:
                ax.bar([col], [float(metrics_table[col].iloc[0])], color="#27ae60")
            ax.set_title(title)
    saved.append(save_fig("09_rankme_c1_c10_global.png"))

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

    global_cols = [c for c in ["rankme_global", "c1_global", "c10_global"] if c in metrics_table.columns]
    if global_cols:
        if "method" in metrics_table.columns and len(metrics_table) > 1:
            global_long = metrics_table.melt(
                id_vars=["method"],
                value_vars=global_cols,
                var_name="metric",
                value_name="value",
            )
            fig_global = px.bar(
                global_long,
                x="method",
                y="value",
                color="metric",
                barmode="group",
                title="Indicateurs géométriques globaux",
                height=500,
            )
        else:
            fig_global = go.Figure(
                go.Bar(
                    x=global_cols,
                    y=[float(metrics_table[c].iloc[0]) for c in global_cols],
                    marker_color=px.colors.qualitative.Set2,
                )
            )
            fig_global.update_layout(title="Indicateurs géométriques globaux", height=450)
        p_global = figures_dir / "09_rankme_c1_c10_global_interactive.html"
        fig_global.write_html(str(p_global), include_plotlyjs="cdn")
        saved.append(p_global)

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
) -> List[Path]:
    """UMAP 2D d'un nuage d'embeddings, coloré par macro (matplotlib + Plotly)."""
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
    "delta_macro_pct",
    "eta2_macro_balanced",
    "rankme_global",
    "c1_global",
    "c10_global",
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
    """Courbes val_delta_macro_pct (et η²) vs epoch, une ligne par fold."""
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
        if "val_delta_macro_pct" in log.columns:
            axes[0].plot(
                log["epoch"],
                log["val_delta_macro_pct"],
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
) -> List[Path]:
    """PCA + t-SNE 2D sur embeddings SCGM projetés (matplotlib + Plotly)."""
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
        )
    )
    pca_pair, tsne_pair = plot_projection_plotly(pca_xy, tsne_xy, sample_df, label_col, figures_dir=figures_dir)
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


def plot_topics_n_units_by_z(
    themes_z: pd.DataFrame,
    *,
    save_fig: Optional[Callable[[str], Path]] = None,
    png_name: str = "topics_n_units_by_z.png",
) -> None:
    """Barplot : ``n_units`` par ``z_id`` (abscisse = composante z)."""
    import matplotlib.pyplot as plt

    if "z_id" not in themes_z.columns or "n_units" not in themes_z.columns:
        print("themes_by_z : colonnes z_id ou n_units absentes")
        return

    df = themes_z.sort_values("z_id")
    fig_w = max(10.0, len(df) * 0.28)
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    ax.bar(df["z_id"].astype(str), df["n_units"].astype(float))
    ax.set_xlabel("Composante z")
    ax.set_ylabel("n_units")
    ax.set_title("Effectif par composante z")
    plt.xticks(rotation=90, fontsize=7)
    plt.tight_layout()
    if save_fig is not None:
        save_fig(png_name)
    else:
        plt.show()
