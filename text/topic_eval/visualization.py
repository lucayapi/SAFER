"""Figures pour la comparaison de qualité thématique."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def bar_metric_by_method(df: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    _ensure_dir(out_path.parent)
    plt.figure(figsize=(8, 4))
    order = df["method"].tolist()
    sns.barplot(data=df, x="method", y=metric, order=order, palette="viridis")
    plt.title(title)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def bubble_quality_plot(df: pd.DataFrame, out_path: Path) -> None:
    _ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(7, 5))
    methods = df["method"].unique()
    colors = sns.color_palette("husl", n_colors=len(methods))
    for i, m in enumerate(methods):
        sub = df.loc[df["method"] == m]
        ax.scatter(
            np.nan_to_num(sub["cv"].to_numpy(dtype=float), nan=0.0),
            np.nan_to_num(sub["topic_diversity"].to_numpy(dtype=float), nan=0.0),
            s=np.clip(sub["coverage"].to_numpy(dtype=float) * 3000, 50, 800),
            c=[colors[i]],
            label=m,
            alpha=0.75,
            edgecolors="k",
        )
    ax.set_xlabel("C_v")
    ax.set_ylabel("Topic Diversity")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Qualité (taille = Coverage)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def topic_size_boxplot(topics_df: pd.DataFrame, out_path: Path) -> None:
    _ensure_dir(out_path.parent)
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=topics_df, x="method", y="n_docs")
    plt.xticks(rotation=15, ha="right")
    plt.ylabel("Taille du topic (n_docs)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def macro_topic_count_heatmap(topics_df: pd.DataFrame, out_path: Path) -> None:
    _ensure_dir(out_path.parent)
    if topics_df.empty or "macro" not in topics_df.columns:
        return
    ct = topics_df.groupby(["method", "macro"]).size().unstack(fill_value=0)
    plt.figure(figsize=(8, 4))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues")
    plt.title("Nombre de topics par macro et méthode")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_all_comparison_figures(
    comparison_metrics_df: pd.DataFrame,
    topics_long: pd.DataFrame,
    figures_dir: str | Path,
) -> None:
    fig_dir = Path(figures_dir)
    _ensure_dir(fig_dir)
    bar_metric_by_method(comparison_metrics_df, "cv", "C_v par méthode", fig_dir / "cv_by_method.png")
    bar_metric_by_method(comparison_metrics_df, "npmi", "NPMI par méthode", fig_dir / "npmi_by_method.png")
    bar_metric_by_method(
        comparison_metrics_df, "topic_diversity", "Topic Diversity par méthode", fig_dir / "topic_diversity_by_method.png"
    )
    bar_metric_by_method(comparison_metrics_df, "redundancy", "Redundancy par méthode", fig_dir / "redundancy_by_method.png")
    bar_metric_by_method(comparison_metrics_df, "coverage", "Coverage par méthode", fig_dir / "coverage_by_method.png")
    bubble_quality_plot(comparison_metrics_df, fig_dir / "quality_bubble_plot.png")
    bar_metric_by_method(comparison_metrics_df, "n_topics", "Nombre de topics par méthode", fig_dir / "n_topics_by_method.png")
    topic_size_boxplot(topics_long, fig_dir / "topic_size_boxplot.png")
    macro_topic_count_heatmap(topics_long, fig_dir / "macro_topic_count_heatmap.png")
