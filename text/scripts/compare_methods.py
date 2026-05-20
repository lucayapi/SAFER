"""Figures et rapport de comparaison des méthodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from metrics.compare_display import slim_geometry_table
from safer_core.paths import ensure_comparisons_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", type=str, default="output")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--table", type=str, default=None)
    p.add_argument(
        "--corpus",
        type=str,
        choices=("btp", "test"),
        default="btp",
        help="Corpus pour figures/rapport (btp ou test métallurgie)",
    )
    return p.parse_args()


def _default_table_path(comp: Path, corpus: str) -> Path:
    tables = comp / "tables"
    if corpus == "test":
        return tables / "embedding_geometry_comparison_test.csv"
    btp = tables / "embedding_geometry_comparison_btp.csv"
    if btp.is_file():
        return btp
    return tables / "embedding_geometry_comparison.csv"


def _plot_bars(df: pd.DataFrame, fig_dir: Path, corpus: str) -> None:
    slim = slim_geometry_table(df)
    prefix = f"{corpus}_"
    metrics = [
        ("eta2_macro_balanced_perc", f"{prefix}eta2_macro_balanced_perc_barplot.png"),
        ("eta2_macro_balanced", f"{prefix}eta2_macro_balanced_barplot.png"),
        ("eta2_weighted", f"{prefix}eta2_weighted_barplot.png"),
        ("rankme_global", f"{prefix}rankme_barplot.png"),
        ("rankme_over_d", f"{prefix}rankme_over_d_barplot.png"),
    ]
    for metric, fname in metrics:
        if metric not in slim.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = slim["method"].astype(str).tolist()
        ax.bar(labels, slim[metric].astype(float))
        ax.set_ylabel(metric)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(f"{corpus} — {metric}")
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=150)
        plt.close(fig)

    x_col = "eta2_macro_balanced_perc" if "eta2_macro_balanced_perc" in slim.columns else "eta2_macro_balanced"
    if x_col in slim.columns and "rankme_global" in slim.columns:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(slim[x_col], slim["rankme_global"])
        for _, r in slim.iterrows():
            ax.annotate(str(r["method"]), (r[x_col], r["rankme_global"]), fontsize=8)
        ax.set_xlabel(x_col)
        ax.set_ylabel("rankme_global")
        ax.set_title(f"{corpus} — eta2 vs RankMe")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{prefix}rankme_vs_eta2.png", dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    comp = ensure_comparisons_dirs()
    out = Path(args.output_dir) if args.output_dir else comp
    table_path = Path(args.table) if args.table else _default_table_path(comp, args.corpus)
    if not table_path.is_file():
        raise FileNotFoundError(f"Tableau manquant : {table_path}. Lancez collect_results.py d'abord.")

    df = pd.read_csv(table_path)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_bars(df, fig_dir, args.corpus)

    report = out / "reports" / f"comparison_report_{args.corpus}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    sort_col = "eta2_macro_balanced_perc" if "eta2_macro_balanced_perc" in df.columns else "eta2_macro_balanced"
    best = df.sort_values(sort_col, ascending=False, na_position="last").head(1) if sort_col in df.columns else df.head(1)
    lines = [
        f"# Rapport de comparaison des embeddings ({args.corpus})",
        "",
        f"Table source : `{table_path}`",
        "",
        f"## Meilleure structuration macro ({sort_col})",
        "",
    ]
    if len(best) and sort_col in best.columns:
        val = best.iloc[0][sort_col]
        lines.append(f"- **{best.iloc[0]['method']}** : {sort_col} = {val:.4f}")
    slim = slim_geometry_table(df)
    lines.extend(["", "## Tableau", "", "```", slim.to_string(index=False), "```"])
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Figures : {fig_dir}")
    print(f"Rapport : {report}")


if __name__ == "__main__":
    main()
