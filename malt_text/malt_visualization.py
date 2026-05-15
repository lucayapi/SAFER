import os
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def ensure_fig_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_barplot(
    frame: pd.DataFrame,
    x: str,
    y: str,
    output_path: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    frame.plot(x=x, y=y, kind="bar", ax=ax, legend=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    output_path: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", xticklabels=col_labels, yticklabels=row_labels, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_pca_scatter(
    coordinates: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    title: str,
    sample_size: int = 10000,
    seed: int = 42,
) -> None:
    if len(coordinates) > sample_size:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(coordinates), size=sample_size, replace=False)
        coordinates = coordinates[indices]
        labels = labels[indices]
    fig, ax = plt.subplots(figsize=(8, 6))
    unique = np.unique(labels)
    for label in unique:
        mask = labels == label
        ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=8, alpha=0.5, label=str(label))
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_anchor_arrows(
    source_points: np.ndarray,
    target_points: np.ndarray,
    labels: Sequence[str],
    output_path: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for index, label in enumerate(labels):
        ax.scatter(source_points[index, 0], source_points[index, 1], c="tab:blue", s=80)
        ax.scatter(target_points[index, 0], target_points[index, 1], c="tab:orange", s=80)
        ax.annotate(
            "",
            xy=(target_points[index, 0], target_points[index, 1]),
            xytext=(source_points[index, 0], source_points[index, 1]),
            arrowprops=dict(arrowstyle="->", color="gray"),
        )
        ax.text(source_points[index, 0], source_points[index, 1], f"{label}s", fontsize=8)
        ax.text(target_points[index, 0], target_points[index, 1], f"{label}t", fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def try_umap_scatter(
    coordinates: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    title: str,
    sample_size: int = 10000,
    seed: int = 42,
) -> bool:
    try:
        import umap
    except ImportError:
        return False
    if len(coordinates) > sample_size:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(coordinates), size=sample_size, replace=False)
        coordinates = coordinates[indices]
        labels = labels[indices]
    reducer = umap.UMAP(n_components=2, random_state=seed)
    embedding = reducer.fit_transform(coordinates)
    save_pca_scatter(embedding, labels, output_path, title, sample_size=len(embedding), seed=seed)
    return True
