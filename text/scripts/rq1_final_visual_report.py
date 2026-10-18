#!/usr/bin/env python3
"""Publication-oriented PDF and figures from the verified RQ1 result tables.

No models or data are recomputed. All plotted values are read from the final
RQ1 CSVs and checked against the six-domain macro row before rendering.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "annotation" / "rq1_final_v1" / "results"
ORDER = ["btp", "metallurgie", "caou_chimie_plas", "bois_textille",
         "service_temporaire", "nicollin"]
NAMES = {
    "btp": "BTP", "metallurgie": "Métallurgie",
    "caou_chimie_plas": "Chimie–plastiques", "bois_textille": "Bois–textile",
    "service_temporaire": "Service temporaire", "nicollin": "Nicollin",
}
COLORS = {"M0": "#8f9eb4", "M1": "#557ca8", "M2": "#e0a14b", "M3": "#176f69"}
ERROR_TYPES = (
    ("span_tp", "Exact match", "gold_units"),
    ("boundary_shift", "Boundary shift", "gold_units"),
    ("fusion", "Merge", "pred_segments"),
    ("fragmented", "Fragmentation", "gold_units"),
    ("missed", "Missed unit", "gold_units"),
    ("spurious", "Spurious unit", "pred_segments"),
)


def load_tables(folder: Path):
    main = pd.read_csv(folder / "table_A_lodo.csv").set_index("target")
    transfer = pd.read_csv(folder / "table_B_transfer_M3.csv").set_index("source")
    comparison = pd.read_csv(folder / "table_C_single_vs_multi.csv").set_index("target")
    ci = pd.read_csv(folder / "bootstrap_paired_2000.csv")
    error_rows = pd.read_csv(folder / "annex_all_metrics.csv")
    error_rows = error_rows[error_rows["experiment"].eq("lodo")].set_index(["target", "model"])
    if set(main.index) != set(ORDER) | {"MACRO_6"} or set(comparison.index) != set(ORDER):
        raise ValueError("Incomplete six-domain RQ1 report")
    if set(transfer.index) != set(ORDER) or set(transfer.columns) != set(ORDER):
        raise ValueError("Incomplete 6×6 single-source transfer matrix")
    if int(main.loc["MACRO_6", "n_test"]) != 2690 or int(main.loc[ORDER, "n_test"].sum()) != 2690:
        raise ValueError("The report no longer uses the 2,690 common test narratives")
    for model in ("M0", "M1", "M2", "M3"):
        if not np.isclose(main.loc[ORDER, f"{model}_exact_f1"].mean(),
                          main.loc["MACRO_6", f"{model}_exact_f1"], atol=1e-12):
            raise ValueError(f"Macro exact F1 mismatch for {model}")
    for target in ORDER:
        if not np.isclose(main.loc[target, "M3_exact_f1"],
                          comparison.loc[target, "lodo_M3_exact_f1"], atol=1e-12):
            raise ValueError(f"LODO score differs between tables: {target}")
        if not np.isnan(transfer.loc[target, target]):
            raise ValueError(f"The single-source diagonal must be N/A: {target}")
        source_values = transfer.loc[[x for x in ORDER if x != target], target].astype(float)
        if not np.isclose(source_values.mean(), comparison.loc[target, "single_source_mean"], atol=1e-12):
            raise ValueError(f"Single-source mean mismatch: {target}")
        if not np.isclose(source_values.max(), comparison.loc[target, "single_source_oracle_max"], atol=1e-12):
            raise ValueError(f"Single-source maximum mismatch: {target}")
    gains = ci[ci["comparison"] == "M3_minus_M2"].set_index("target")
    if set(gains.index) != set(ORDER) | {"MACRO_6"}:
        raise ValueError("Missing paired bootstrap intervals")
    if not np.allclose(gains.loc[ORDER, "delta_exact_f1"].to_numpy(),
                       main.loc[ORDER, "gain_M3_minus_M2"].to_numpy()):
        raise ValueError("Gain and bootstrap tables differ")
    if set(error_rows.index) != {(target, model) for target in ORDER
                                for model in ("M0", "M1", "M2", "M3")}:
        raise ValueError("Incomplete LODO error diagnostics")
    for target in ORDER:
        for model in ("M0", "M1", "M2", "M3"):
            item = error_rows.loc[(target, model)]
            if (int(item["n_test"]) != int(main.loc[target, "n_test"]) or
                not np.isclose(item["soft_f1"], main.loc[target, f"{model}_soft_f1"]) or
                not np.isclose(item["soft75_f1"], main.loc[target, f"{model}_soft75_f1"]) or
                not np.isclose(item["exact_f1"], main.loc[target, f"{model}_exact_f1"])):
                raise ValueError(f"Diagnostic table mismatch: {target}/{model}")
        if error_rows.loc[target, "gold_units"].nunique() != 1:
            raise ValueError(f"Different GOLD unit counts on {target}")
    return main, transfer, comparison, gains, error_rows


def style(ax, *, y_max: float = 0.75):
    ax.set_ylim(0, y_max)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e0e5eb", linewidth=0.7)
    ax.set_axisbelow(True)


def save_figure(fig, folder: Path, stem: str, pdf: PdfPages) -> None:
    fig.savefig(folder / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(folder / f"{stem}.png", dpi=180, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def title_page(main: pd.DataFrame, gains: pd.DataFrame, comparison: pd.DataFrame):
    fig = plt.figure(figsize=(11.7, 8.3), layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[0.9, 1.3, 0.9])
    ax = fig.add_subplot(gs[0])
    ax.axis("off")
    ax.text(0, 0.92, "RQ1 · Segmentation factuelle", fontsize=23, fontweight="bold", color="#14334b")
    ax.text(0, 0.62, "GOLD narrative_v3 · 2 690 récits de test communs · six domaines", fontsize=12)
    ax.text(0, 0.38, "M0 PySBD  ·  M1 SaT initial  ·  M2 SaT calibré  ·  M3 SaT + LoRA multisource", fontsize=11)
    ax.text(0, 0.12, "Métrique principale : F1 des segments exacts, espaces périphériques neutralisés.", fontsize=10)
    table_ax = fig.add_subplot(gs[1])
    table_ax.axis("off")
    rows = []
    for target in ORDER:
        row = main.loc[target]
        rows.append([NAMES[target], str(int(row["n_test"])),
                     *[f"{row[f'{m}_exact_f1']:.3f}" for m in ("M0", "M1", "M2", "M3")],
                     f"{row['gain_M3_minus_M2']:+.3f}"])
    macro = main.loc["MACRO_6"]
    rows.append(["Moyenne macro", "2 690", *[f"{macro[f'{m}_exact_f1']:.3f}"
                for m in ("M0", "M1", "M2", "M3")], f"{macro['gain_M3_minus_M2']:+.3f}"])
    table = table_ax.table(cellText=rows,
                           colLabels=["Domaine cible", "n", "M0", "M1", "M2", "M3", "M3−M2"],
                           loc="center", cellLoc="center", colLoc="center",
                           colWidths=[0.28, 0.09, 0.105, 0.105, 0.105, 0.105, 0.13])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d5dce4")
        if r == 0:
            cell.set_facecolor("#14334b")
            cell.set_text_props(color="white", weight="bold")
        elif r == len(rows):
            cell.set_facecolor("#e7f2ef")
            cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f3f6f9")
    note = fig.add_subplot(gs[2])
    note.axis("off")
    ci = gains.loc["MACRO_6"]
    note.text(0, 0.82,
              f"Gain M3−M2 : {ci['delta_exact_f1']:+.3f}  ·  IC 95 % apparié "
              f"[{ci['ci95_low']:+.3f} ; {ci['ci95_high']:+.3f}]", fontsize=12,
              color="#176f69", weight="bold")
    note.text(0, 0.60, "Les six gains M3−M2 sont positifs ; leur amplitude varie selon le domaine.", fontsize=10)
    single_mean = comparison.loc[ORDER, "single_source_mean"].mean()
    multi_mean = comparison.loc[ORDER, "lodo_M3_exact_f1"].mean()
    note.text(0, 0.35,
              f"Multisource {multi_mean:.3f} contre moyenne monosource {single_mean:.3f} "
              f"(écart {multi_mean - single_mean:+.3f}, descriptif).", fontsize=10)
    note.text(0, 0.12, "Limites : une graine LoRA ; GOLD confirmé à partir de préannotations IA, sans double annotation indépendante.", fontsize=9)
    return fig


def multi_domain_figure(main: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11.7, 7.0), layout="constrained")
    x = np.arange(6)
    width = 0.19
    for index, model in enumerate(("M0", "M1", "M2", "M3")):
        values = main.loc[ORDER, f"{model}_exact_f1"].to_numpy(float)
        ax.bar(x + (index - 1.5) * width, values, width=width, color=COLORS[model],
               label=model, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x, [NAMES[target] for target in ORDER], rotation=20, ha="right")
    ax.set_ylabel("F1 des segments exacts")
    ax.set_title("Multisource LODO : mêmes récits de test pour les quatre méthodes", fontsize=15, weight="bold")
    ax.legend(ncol=4, frameon=False, loc="upper left")
    style(ax, y_max=0.72)
    return fig


def gain_figure(gains: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11.7, 7.0), layout="constrained")
    y = np.arange(6)
    points = gains.loc[ORDER, "delta_exact_f1"].to_numpy(float)
    left = points - gains.loc[ORDER, "ci95_low"].to_numpy(float)
    right = gains.loc[ORDER, "ci95_high"].to_numpy(float) - points
    ax.errorbar(points, y, xerr=[left, right], fmt="o", color=COLORS["M3"],
                ecolor="#48737b", elinewidth=2, capsize=4, markersize=8)
    ax.axvline(0, color="#a23b3b", linewidth=1)
    ax.set_yticks(y, [NAMES[target] for target in ORDER])
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 0.55)
    ax.set_xlabel("Gain de F1 exact : M3 − M2")
    ax.set_title("Gain de LoRA sur SaT calibré, avec IC 95 % appariés", fontsize=15, weight="bold")
    ax.grid(axis="x", color="#e0e5eb")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def complementary_figure(main: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(11.7, 8.3), layout="constrained")
    x = np.arange(6)
    width = 0.19
    for ax, metric, title in (
        (axes[0], "boundary_f1", "F1 des frontières, espaces neutralisés"),
        (axes[1], "soft_f1", "Soft Span F1, IoU pondéré et appariement un-à-un"),
    ):
        for index, model in enumerate(("M0", "M1", "M2", "M3")):
            values = main.loc[ORDER, f"{model}_{metric}"].to_numpy(float)
            ax.bar(x + (index - 1.5) * width, values, width=width,
                   color=COLORS[model], label=model, edgecolor="white", linewidth=0.4)
        ax.set_xticks(x, [NAMES[target] for target in ORDER], rotation=18, ha="right")
        ax.set_ylabel("F1")
        ax.set_title(title, fontsize=13, weight="bold")
        style(ax, y_max=0.9)
    axes[0].legend(ncol=4, frameon=False, loc="upper left")
    return fig


def soft_explanation_figure(main: pd.DataFrame, metrics: pd.DataFrame):
    fig = plt.figure(figsize=(11.7, 8.3), layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[0.35, 1.8, 1.05])
    title = fig.add_subplot(gs[0]); title.axis("off")
    title.text(0, 0.75, "Récupération partielle des unités factuelles", fontsize=18, weight="bold")
    table_ax = fig.add_subplot(gs[1]); table_ax.axis("off")
    rows = []
    for target in ORDER + ["MACRO_6"]:
        row = main.loc[target]
        m2, m3 = row["M2_soft_f1"], row["M3_soft_f1"]
        rows.append([NAMES.get(target, "Moyenne macro"), f"{m2:.3f}", f"{m3:.3f}",
                     f"{m3 - m2:+.3f}", f"{row['M2_soft75_f1']:.3f}",
                     f"{row['M3_soft75_f1']:.3f}"])
    table = table_ax.table(cellText=rows,
                           colLabels=["Domaine cible", "M2 soft", "M3 soft", "Gain", "M2 IoU ≥ 0,75", "M3 IoU ≥ 0,75"],
                           loc="center", cellLoc="center", colLoc="center",
                           colWidths=[0.27, 0.12, 0.12, 0.12, 0.18, 0.18])
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.75)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d5dce4")
        if r == 0:
            cell.set_facecolor("#14334b"); cell.set_text_props(color="white", weight="bold")
        elif r == len(rows):
            cell.set_facecolor("#e7f2ef"); cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f3f6f9")
    note = fig.add_subplot(gs[2]); note.axis("off")
    p2 = metrics.xs("M2", level="model")["soft_precision"].mean()
    r2 = metrics.xs("M2", level="model")["soft_recall"].mean()
    p3 = metrics.xs("M3", level="model")["soft_precision"].mean()
    r3 = metrics.xs("M3", level="model")["soft_recall"].mean()
    note.text(0, 0.88, "Chaque prédiction est appariée à au plus une unité GOLD ; le crédit est son IoU, entre 0 et 1.", fontsize=10)
    note.text(0, 0.68, "Soft F1 = 2 × somme des crédits / (nombre prédit + nombre GOLD). Espaces périphériques neutralisés.", fontsize=10)
    note.text(0, 0.48, "La variante IoU ≥ 0,75 garde le crédit proportionnel, mais écarte les paires sous ce seuil.", fontsize=10)
    note.text(0, 0.28, f"Précision / rappel soft moyens : M2 {p2:.3f} / {r2:.3f} ; M3 {p3:.3f} / {r3:.3f}.", fontsize=10)
    note.text(0, 0.08, "M3 retrouve davantage de contenu, mais le crédit partiel ne garantit ni les bornes exactes ni le bon rôle.", fontsize=10)
    return fig


def error_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in ORDER + ["TOUS"]:
        for model in ("M2", "M3"):
            selected = (metrics.loc[(target, model)].to_frame().T if target != "TOUS"
                        else metrics.xs(model, level="model").loc[ORDER])
            for key, label, denominator_key in ERROR_TYPES:
                count = int(selected[key].sum())
                denominator = int(selected[denominator_key].sum())
                if count < 0 or denominator <= 0 or count > denominator:
                    raise ValueError(f"Invalid error rate: {target}/{model}/{key}")
                rows.append({"target": target, "model": model, "error_type": key,
                             "label": label, "count": count, "denominator": denominator,
                             "rate": count / denominator})
    return pd.DataFrame(rows)


def error_overview_figure(summary: pd.DataFrame):
    fig = plt.figure(figsize=(11.7, 8.3), layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[0.35, 2.2, 1.35])
    title = fig.add_subplot(gs[0]); title.axis("off")
    title.text(0, 0.75, "Correspondances et erreurs · tous les récits de test", fontsize=18, weight="bold")
    ax = fig.add_subplot(gs[1])
    keys = [item[0] for item in ERROR_TYPES]
    labels = [item[1].replace(" ", "\n", 1) for item in ERROR_TYPES]
    x = np.arange(len(keys)); width = 0.36
    for offset, model in ((-0.5, "M2"), (0.5, "M3")):
        frame = summary[summary["target"].eq("TOUS") & summary["model"].eq(model)].set_index("error_type")
        values = frame.loc[keys, "rate"].to_numpy(float)
        bars = ax.bar(x + offset * width, values, width, label=model, color=COLORS[model])
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=3, fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("Taux sur le dénominateur indiqué")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.grid(axis="y", color="#e0e5eb"); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    note = fig.add_subplot(gs[2]); note.axis("off")
    note.text(0, 0.90, "Lecture : M3 réduit fortement les fusions et les unités manquées, mais accroît les fragmentations et les décalages IoU ≥ 0,75.", fontsize=10, weight="bold")
    note.text(0, 0.73, "Exact match, boundary shift, fragmentation et missed unit : dénominateur = 21 850 unités GOLD.", fontsize=10)
    note.text(0, 0.56, "Merge et spurious unit : dénominateur = segments prédits (M2 : 19 731 ; M3 : 22 614).", fontsize=10)
    note.text(0, 0.39, "Shift : un seul segment recouvrant l'unité, IoU ≥ 0,75, mais bornes différentes ; merge/fragment : tout chevauchement.", fontsize=10)
    note.text(0, 0.22, "Missed/spurious : aucune paire avec IoU ≥ 0,75 ; un chevauchement plus faible reste possible.", fontsize=10)
    note.text(0, 0.05, "Diagnostics descriptifs non exclusifs : ces taux ne doivent pas être additionnés.", fontsize=10)
    return fig


def error_domain_figure(summary: pd.DataFrame):
    keys = [item[0] for item in ERROR_TYPES]
    values = np.zeros((len(ORDER), len(keys)))
    for i, target in enumerate(ORDER):
        for j, key in enumerate(keys):
            m2 = summary[(summary.target.eq(target)) & summary.model.eq("M2") & summary.error_type.eq(key)].iloc[0]
            m3 = summary[(summary.target.eq(target)) & summary.model.eq("M3") & summary.error_type.eq(key)].iloc[0]
            values[i, j] = 100 * (m3["rate"] - m2["rate"])
    scale = max(10, float(np.ceil(np.abs(values).max() / 10) * 10))
    fig, ax = plt.subplots(figsize=(11.7, 8.3), layout="constrained")
    image = ax.imshow(values, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-scale, vmax=scale), aspect="auto")
    ax.set_xticks(range(len(keys)), [item[1].replace(" ", "\n", 1) for item in ERROR_TYPES])
    ax.set_yticks(range(len(ORDER)), [NAMES[target] for target in ORDER])
    for i in range(len(ORDER)):
        for j in range(len(keys)):
            ax.text(j, i, f"{values[i, j]:+.1f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(values[i, j]) > scale * 0.6 else "#173042")
    ax.set_title("Évolution des diagnostics par domaine : M3 − M2", fontsize=16, weight="bold")
    ax.set_xlabel("Catégorie diagnostique · variation en points de pourcentage")
    fig.colorbar(image, ax=ax, shrink=0.75, label="M3 − M2 (points)")
    return fig


def transfer_figure(transfer: pd.DataFrame):
    values = transfer.loc[ORDER, ORDER].to_numpy(float)
    mask = np.ma.masked_invalid(values)
    fig, ax = plt.subplots(figsize=(10.8, 8.0), layout="constrained")
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#e8e8e8")
    image = ax.imshow(mask, cmap=cmap, vmin=0.25, vmax=0.70, aspect="equal")
    for i in range(6):
        for j in range(6):
            label = "N/A" if i == j else f"{values[i, j]:.3f}"
            ax.text(j, i, label, ha="center", va="center",
                    color="white" if i != j and values[i, j] > 0.55 else "#173042",
                    fontsize=10, weight="bold" if i == j else "normal")
    ax.set_xticks(range(6), [NAMES[x] for x in ORDER], rotation=30, ha="right")
    ax.set_yticks(range(6), [NAMES[x] for x in ORDER])
    ax.set_xlabel("Corpus de test")
    ax.set_ylabel("Corpus d'entraînement unique")
    ax.set_title("Transfert monosource : F1 exact de chaque modèle adapté", fontsize=15, weight="bold")
    fig.colorbar(image, ax=ax, shrink=0.75, label="F1 des segments exacts")
    return fig


def comparison_figure(comp: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11.7, 7.1), layout="constrained")
    y = np.arange(6)
    data = comp.loc[ORDER]
    x_lodo = data["lodo_M3_exact_f1"].to_numpy(float)
    x_mean = data["single_source_mean"].to_numpy(float)
    x_oracle = data["single_source_oracle_max"].to_numpy(float)
    ax.barh(y - 0.22, x_lodo, height=0.21, color=COLORS["M3"], label="M3 multisource LODO")
    ax.barh(y, x_mean, height=0.21, color="#7d98b4", label="Moyenne des cinq monosources")
    ax.barh(y + 0.22, x_oracle, height=0.21, color="#ddb15b", label="Maximum monosource (oracle)")
    ax.set_yticks(y, [NAMES[target] for target in ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.76)
    ax.set_xlabel("F1 des segments exacts sur le corpus cible")
    ax.set_title("Multisource contre transfert monosource", fontsize=15, weight="bold")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3)
    ax.grid(axis="x", color="#e0e5eb")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def delta_figure(comp: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11.7, 6.5), layout="constrained")
    delta = (comp.loc[ORDER, "lodo_M3_exact_f1"] - comp.loc[ORDER, "single_source_mean"]).to_numpy(float)
    y = np.arange(6)
    ax.barh(y, delta, color=["#176f69" if value >= 0 else "#c8674b" for value in delta])
    for index, value in enumerate(delta):
        ax.text(value + (0.004 if value >= 0 else -0.004), index, f"{value:+.3f}",
                ha="left" if value >= 0 else "right", va="center", fontsize=10)
    ax.axvline(0, color="#173042", linewidth=1)
    ax.set_yticks(y, [NAMES[target] for target in ORDER])
    ax.invert_yaxis()
    ax.set_xlim(-0.16, 0.09)
    ax.set_xlabel("F1 exact : M3 multisource − moyenne des cinq monosources")
    ax.set_title("Le multisource n'est pas supérieur sur tous les domaines", fontsize=15, weight="bold")
    ax.grid(axis="x", color="#e0e5eb")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def render(folder: Path) -> Path:
    main, transfer, comp, gains, metrics = load_tables(folder)
    errors = error_summary(metrics)
    errors.to_csv(folder / "error_taxonomy_summary.csv", index=False, encoding="utf-8-sig")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 15, "savefig.facecolor": "white"})
    dest = folder / "RQ1_resultats_graphiques.pdf"
    with PdfPages(dest, metadata={"Title": "RQ1 — segmentation factuelle",
                                  "Author": "SAFER research analysis"}) as pdf:
        fig = title_page(main, gains, comp)
        fig.savefig(folder / "RQ1_synthese.png", dpi=180, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        save_figure(multi_domain_figure(main), folder, "figure_multisource_M0_M3", pdf)
        save_figure(gain_figure(gains), folder, "figure_gain_multisource_M3_M2", pdf)
        save_figure(complementary_figure(main), folder, "figure_frontieres_soft_multisource", pdf)
        save_figure(soft_explanation_figure(main, metrics), folder, "figure_soft_span_explique", pdf)
        save_figure(error_overview_figure(errors), folder, "figure_typologie_erreurs", pdf)
        save_figure(error_domain_figure(errors), folder, "figure_erreurs_par_domaine", pdf)
        save_figure(transfer_figure(transfer), folder, "figure_matrice_monosource", pdf)
        save_figure(comparison_figure(comp), folder, "figure_multisource_vs_monosource", pdf)
        save_figure(delta_figure(comp), folder, "figure_delta_multi_mono", pdf)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    print(render(args.results.resolve()))


if __name__ == "__main__":
    main()
