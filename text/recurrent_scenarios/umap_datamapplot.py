"""DataMapPlot-style 2-D topic maps for recurrent-scenario results notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from manuscript_reporting import coalesce_text, sanitize_label_text


def resolve_llm_topic_label(label_row: pd.Series | None, *, topic_id: str = "", topic_index: int | None = None) -> str:
    """Return the LLM label when available; otherwise the topic id."""
    if label_row is not None and isinstance(label_row, pd.Series):
        label = coalesce_text(label_row.get("llm_label"), default="")
        if label:
            return label
    if topic_id:
        return topic_id
    if topic_index is not None and int(topic_index) >= 0:
        return f"Topic {int(topic_index)}"
    return "Noise"


def build_llm_label_lookup(theme_labels: pd.DataFrame, *, role: str, configuration_id: str) -> dict[str, str]:
    """Map ``topic_id`` to LLM labels for one selected partition."""
    subset = theme_labels[
        theme_labels["role"].astype(str).eq(role)
        & theme_labels["configuration_id"].astype(str).eq(str(configuration_id))
    ]
    if subset.empty:
        return {}
    lookup: dict[str, str] = {}
    for topic_id, group in subset.groupby("topic_id"):
        lookup[str(topic_id)] = resolve_llm_topic_label(group.iloc[0], topic_id=str(topic_id))
    return lookup


def build_topic_label_array(
    topic_ids: pd.Series | np.ndarray,
    label_lookup: Mapping[str, str],
    *,
    noise_label: str = "Noise",
) -> np.ndarray:
    """Per-point cluster labels for DataMapPlot (noise uses ``noise_label``)."""
    labels = []
    for topic_id in np.asarray(topic_ids).astype(str):
        topic_id = topic_id.strip()
        if not topic_id:
            labels.append(noise_label)
        else:
            labels.append(str(label_lookup.get(topic_id, topic_id)))
    return np.asarray(labels, dtype=object)


def _apply_tight_crop(axis, coordinates: np.ndarray, *, padding_fraction: float = 0.10) -> None:
    if coordinates.size == 0:
        return
    x_values = coordinates[:, 0]
    y_values = coordinates[:, 1]
    x_span = float(x_values.max() - x_values.min())
    y_span = float(y_values.max() - y_values.min())
    x_pad = max(x_span * padding_fraction, 0.5)
    y_pad = max(y_span * padding_fraction, 0.5)
    axis.set_xlim(float(x_values.min()) - x_pad, float(x_values.max()) + x_pad)
    axis.set_ylim(float(y_values.min()) - y_pad, float(y_values.max()) + y_pad)


def plot_role_topic_datamap(
    coordinates: np.ndarray,
    labels: np.ndarray,
    *,
    title: str = "",
    output_path: Path | None = None,
    figsize: tuple[float, float] = (14.0, 12.0),
    dpi: float = 300.0,
    label_wrap_width: int = 22,
    label_font_size: float = 9.0,
    dynamic_label_size: bool = False,
    noise_label: str = "Noise",
    noise_color: str = "#d9d9d9",
    title_fontsize: float = 11.0,
    tight_crop: bool = False,
    show_noise_note: bool = True,
) -> object:
    """Render a publication-style DataMapPlot figure (glow + exterior labels)."""
    import datamapplot as dmp
    import matplotlib.pyplot as plt

    coordinates = np.asarray(coordinates, dtype=float)
    labels = np.asarray(labels, dtype=object)
    figure, axis = dmp.create_plot(
        coordinates,
        labels,
        title="",
        sub_title=None,
        noise_label=noise_label,
        noise_color=noise_color,
        color_label_text=True,
        color_label_arrows=True,
        label_wrap_width=int(label_wrap_width),
        label_font_size=float(label_font_size),
        figsize=figsize,
        dpi=float(dpi),
        dynamic_label_size=bool(dynamic_label_size),
        point_size=1.0,
        arrowprops={"arrowstyle": "wedge, tail_width=0.35, shrink_factor=0.25"},
    )
    if title:
        axis.set_title(title, fontsize=title_fontsize, loc="left", pad=6)
    if tight_crop:
        _apply_tight_crop(axis, coordinates)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    has_noise = bool(np.any(labels.astype(str) == noise_label))
    if show_noise_note and has_noise:
        figure.text(
            0.5,
            0.01,
            "Grey points indicate observations classified as noise by HDBSCAN.",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    figure.tight_layout(rect=(0, 0.015 if show_noise_note and has_noise else 0, 1, 1), pad=0.05)
    if output_path is not None:
        output_path = Path(output_path)
        from manuscript_reporting import save_manuscript_figure

        save_manuscript_figure(figure, output_path, dpi=int(dpi))
    return figure


__all__ = [
    "build_llm_label_lookup",
    "build_topic_label_array",
    "plot_role_topic_datamap",
    "resolve_llm_topic_label",
]
