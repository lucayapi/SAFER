"""Figures et rapport de comparaison des méthodes."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from safer_core.paths import ensure_comparisons_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", type=str, default="resultats")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--table", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    comp = ensure_comparisons_dirs()
    out = Path(args.output_dir) if args.output_dir else comp
    table_path = Path(args.table) if args.table else comp / "tables" / "embedding_geometry_comparison.csv"
    if not table_path.is_file():
        raise FileNotFoundError(f"Tableau manquant : {table_path}. Lancez collect_results.py d'abord.")

    df = pd.read_csv(table_path)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for metric, fname in [
        ("eta2_macro_balanced", "eta2_macro_balanced_barplot.png"),
        ("eta2_weighted", "eta2_weighted_barplot.png"),
        ("rankme_global", "rankme_barplot.png"),
    ]:
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = df["method"].astype(str).tolist()
        ax.bar(labels, df[metric].astype(float))
        ax.set_ylabel(metric)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(metric)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=150)
        plt.close(fig)

    if "eta2_macro_balanced" in df.columns and "rankme_global" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(df["eta2_macro_balanced"], df["rankme_global"])
        for _, r in df.iterrows():
            ax.annotate(str(r["method"]), (r["eta2_macro_balanced"], r["rankme_global"]), fontsize=8)
        ax.set_xlabel("eta2_macro_balanced")
        ax.set_ylabel("rankme_global")
        ax.set_title("eta2 vs RankMe")
        fig.tight_layout()
        fig.savefig(fig_dir / "rankme_vs_eta2.png", dpi=150)
        plt.close(fig)

    report = out / "reports" / "comparison_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    best = df.sort_values("eta2_macro_balanced", ascending=False).head(1)
    lines = [
        "# Rapport de comparaison des embeddings",
        "",
        f"Table source : `{table_path}`",
        "",
        "## Meilleure structuration macro (eta2_macro_balanced)",
        "",
    ]
    if len(best):
        lines.append(f"- **{best.iloc[0]['method']}** : eta2_macro_balanced = {best.iloc[0]['eta2_macro_balanced']:.4f}")
    lines.extend(["", "## Tableau", "", "```", df.to_string(index=False), "```"])
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Figures : {fig_dir}")
    print(f"Rapport : {report}")


if __name__ == "__main__":
    main()
