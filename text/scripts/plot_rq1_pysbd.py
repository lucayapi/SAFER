#!/usr/bin/env python3
r"""Plot the PySBD RQ1 baseline and its paired differences from SaT M0.

Run from the SAFER root:
    .\text\.venv\Scripts\python.exe text\scripts\plot_rq1_pysbd.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "annotation" / "rq1_pysbd_v3_20261017"
LABELS = {
    "caou_chimie_plas": "Chimie-plastiques",
    "btp": "BTP",
    "metallurgie": "Métallurgie",
    "bois_textille": "Bois-textile",
    "nicollin": "Nicollin",
    "service_temporaire": "Service temporaire",
}
MACRO = "MACRO_6_CORPUS"
PANELS = (
    ("span", "Segments exacts", "#37699a"),
    ("boundary", "Frontières", "#23957d"),
)


def read_results(folder: Path) -> tuple[dict[str, dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    with (folder / "comparison_M0_PySBD.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["subset"] == "all"]
    by_target = {row["target"]: row for row in rows}
    if len(rows) != len(by_target) or set(by_target) != set(LABELS) | {MACRO}:
        raise ValueError("Le tableau doit contenir exactement six corpus et leur moyenne macro.")

    with (folder / "gain_ci95.csv").open(encoding="utf-8-sig", newline="") as stream:
        ci_rows = list(csv.DictReader(stream))
    ci = {(row["target"], row["metric"]): row for row in ci_rows}
    if len(ci) != len(ci_rows) or set(ci) != {
        (target, f"{kind}_f1") for target in by_target for kind in ("span", "boundary")
    }:
        raise ValueError("Intervalles de confiance incomplets ou dupliqués.")

    for kind, _, _ in PANELS:
        gain_key = f"gain_{kind}_f1"
        for target, row in by_target.items():
            m0, pysbd, gain = (float(row[key]) for key in
                               (f"M0_{kind}_f1", f"PySBD_{kind}_f1", gain_key))
            if not math.isclose(pysbd - m0, gain, abs_tol=1e-12):
                raise ValueError(f"Gain incohérent : {target}, {kind}.")
            interval = ci[target, f"{kind}_f1"]
            low, high = float(interval["ci95_low"]), float(interval["ci95_high"])
            if not low <= high:
                raise ValueError(f"IC inversé : {target}, {kind}.")
        mean = sum(float(by_target[name][gain_key]) for name in LABELS) / len(LABELS)
        if not math.isclose(mean, float(by_target[MACRO][gain_key]), abs_tol=1e-12):
            raise ValueError(f"Moyenne macro incohérente : {kind}.")
    return by_target, ci


def save(fig: plt.Figure, folder: Path, stem: str) -> None:
    for extension in ("png", "pdf"):
        path = folder / f"{stem}.{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        print(path)
    plt.close(fig)


def plot_pysbd(folder: Path, rows: dict[str, dict[str, str]]) -> None:
    names = list(LABELS)
    positions = list(range(len(names)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True, constrained_layout=True)
    for ax, (kind, title, color) in zip(axes, PANELS):
        values = [float(rows[name][f"PySBD_{kind}_f1"]) for name in names]
        macro = float(rows[MACRO][f"PySBD_{kind}_f1"])
        ax.barh(positions, values, height=0.62, color=color)
        ax.axvline(macro, color="#343434", linestyle="--", linewidth=1.3)
        for y, value in zip(positions, values):
            ax.text(value + 0.012, y, f"{value:.3f}", va="center", fontsize=9)
        ax.set_xlim(0, 0.76)
        ax.set_xticks([i / 10 for i in range(8)])
        ax.set_xlabel("F1")
        ax.set_title(f"{title} (macro : {macro:.3f})", fontweight="bold")
        ax.grid(axis="x", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(positions, [LABELS[name] for name in names])
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", left=False)
    fig.suptitle("PySBD : résultats par corpus — trait pointillé : moyenne macro",
                 fontsize=13, fontweight="bold")
    save(fig, folder, "figure_PySBD_f1_par_corpus")


def plot_gain(folder: Path, rows: dict[str, dict[str, str]],
              ci: dict[tuple[str, str], dict[str, str]]) -> None:
    names = list(LABELS)
    positions = list(range(len(names)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True, constrained_layout=True)
    for ax, (kind, title, color) in zip(axes, PANELS):
        values = [float(rows[name][f"gain_{kind}_f1"]) for name in names]
        intervals = [ci[name, f"{kind}_f1"] for name in names]
        lows = [float(item["ci95_low"]) for item in intervals]
        highs = [float(item["ci95_high"]) for item in intervals]
        macro = float(rows[MACRO][f"gain_{kind}_f1"])
        ax.barh(positions, values, height=0.52, color=color)
        ax.errorbar(values, positions,
                    xerr=[[value - low for value, low in zip(values, lows)],
                          [high - value for value, high in zip(values, highs)]],
                    fmt="none", ecolor="#292929", elinewidth=1.2, capsize=3)
        ax.axvline(0, color="#343434", linewidth=1.1)
        left = min(lows + [0])
        right = max(highs + [0])
        width = right - left
        ax.set_xlim(left - width * 0.23, right + width * 0.47)
        for y, value, low in zip(positions, values, lows):
            ax.text(low - width * 0.035, y, f"{value:+.4f}", ha="right",
                    va="center", fontsize=9)
        ax.set_xlabel("Différence de F1 (PySBD − M0)")
        ax.set_title(f"{title} (macro : {macro:+.4f})", fontweight="bold")
        ax.grid(axis="x", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(positions, [LABELS[name] for name in names])
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", left=False)
    fig.suptitle("Gain de PySBD par rapport à SaT M0 — traits : IC 95 % appariés",
                 fontsize=13, fontweight="bold")
    save(fig, folder, "figure_gain_PySBD_moins_M0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    folder = args.results_dir.resolve()
    rows, ci = read_results(folder)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    plot_pysbd(folder, rows)
    plot_gain(folder, rows, ci)


if __name__ == "__main__":
    main()
