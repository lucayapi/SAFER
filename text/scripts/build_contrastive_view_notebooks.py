"""Generate separate result viewers for the three contrastive methods."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO / "notebooks"
METHODS = (
    ("batch_triplet", "Batch Triplet"),
    ("supcon", "SupCon"),
    ("softtriple", "SoftTriple"),
)


def _cell(source: str, cell_type: str = "code", cell_id: str | None = None) -> dict:
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")],
    }
    if cell_type == "code":
        cell.update({"outputs": [], "execution_count": None})
    if cell_id:
        cell["id"] = cell_id
    return cell


def _notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


CONFIG_CODE = r'''from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from IPython.display import display

from safer_core.brand_style import apply_matplotlib_brand

ROOT = TEXT_ROOT
METHOD_KEY = "__METHOD_KEY__"
DISPLAY_NAME = "__DISPLAY_NAME__"
CONFIG_PATH = ROOT / "configs" / "methods" / f"{METHOD_KEY}.yaml"
METHOD_CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
TEST_CORPORA = list(
    (METHOD_CONFIG.get("data") or {}).get("test_corpora")
    or METHOD_CONFIG.get("test_corpora")
    or ["metallurgie", "caou", "nicollin"]
)
RESULTS_DIR = ROOT / "output" / METHOD_KEY / "macro_ft_tuning"
FIGURES_DIR = RESULTS_DIR / "figures_notebook"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ROLE_COLORS = {
    "A0": "#0072B2",
    "A1": "#E69F00",
    "B": "#009E73",
    "C": "#D55E00",
}
CORPUS_LABELS = {
    "btp": "Construction",
    "metallurgie": "Metallurgy",
    "caou": "Chemistry-plastics",
    "nicollin": "Company",
}
TSNE_VARIANT_ID = None
TSNE_CORPUS = "btp"
TSNE_MAX_POINTS = None
TSNE_SEED = 42

apply_matplotlib_brand()
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 96

print("Method:", DISPLAY_NAME)
print("Results:", RESULTS_DIR)
print("Figures:", FIGURES_DIR)
'''

TABLE_CODE = r'''results = pd.read_csv(RESULTS_DIR / "results_summary.csv")
if results.empty:
    raise FileNotFoundError(f"No contrastive results found in {RESULTS_DIR}")

metric_table = pd.DataFrame()
metric_table["Model"] = (
    results["encoder_scope"].astype(str)
    + " / projector "
    + results["projector"].astype(str)
)
metric_table["CV BA ± std (BTP)"] = results.apply(
    lambda row: f"{float(row['cv_ba_mean']):.3f} ± {float(row['cv_ba_std']):.3f}",
    axis=1,
)
for corpus_id in TEST_CORPORA:
    metric_table[f"BA {CORPUS_LABELS.get(corpus_id, corpus_id)}"] = results[
        f"ba_ood_{corpus_id}"
    ]
metric_table["BA OOD Avg"] = results["ba_ood_avg"]
metric_table["BA OOD Worst"] = results["ba_ood_worst"]
display(
    metric_table.style.format(
        {
            column: "{:.3f}"
            for column in metric_table.columns
            if column.startswith("BA ")
        },
        na_rep="—",
    )
)

best_row = results.sort_values("ba_ood_avg", ascending=False).iloc[0]
BEST_VARIANT_ID = str(best_row["combo_id"])
print("Best variant selected by BA OOD average:", BEST_VARIANT_ID)
print(
    "Best model:",
    f"{best_row['encoder_scope']} / projector {best_row['projector']}",
)
'''

TSNE_CODE = r'''from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

selected_variant_id = TSNE_VARIANT_ID or BEST_VARIANT_ID
variant_dir = RESULTS_DIR / "combos" / str(selected_variant_id)
embedding_path = variant_dir / "embeddings" / f"projected_{TSNE_CORPUS}.npy"
metadata_path = variant_dir / "embeddings" / f"projected_{TSNE_CORPUS}_metadata.csv"

if not embedding_path.is_file() or not metadata_path.is_file():
    print("Projected embeddings missing:", embedding_path)
else:
    tsne_embeddings = np.load(embedding_path)
    tsne_metadata = pd.read_csv(metadata_path)
    if len(tsne_embeddings) != len(tsne_metadata):
        raise ValueError("Projected embeddings and metadata are not aligned")
    role_column = "pred_label" if "pred_label" in tsne_metadata.columns else "true_macro"
    if role_column not in tsne_metadata.columns:
        raise KeyError("True accident-process role column is missing")

    if TSNE_MAX_POINTS is not None and len(tsne_embeddings) > int(TSNE_MAX_POINTS):
        rng = np.random.default_rng(TSNE_SEED)
        selected_indices = np.sort(
            rng.choice(len(tsne_embeddings), size=int(TSNE_MAX_POINTS), replace=False)
        )
        tsne_embeddings = tsne_embeddings[selected_indices]
        tsne_metadata = tsne_metadata.iloc[selected_indices].reset_index(drop=True)
        points_note = f"sample of {len(tsne_metadata):,} points"
    else:
        points_note = f"all {len(tsne_metadata):,} points"

    pca_components = min(50, tsne_embeddings.shape[1], len(tsne_embeddings) - 1)
    tsne_input = PCA(
        n_components=pca_components,
        random_state=TSNE_SEED,
    ).fit_transform(tsne_embeddings)
    perplexity = min(30, max(5, (len(tsne_input) - 1) // 3))
    tsne_coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=TSNE_SEED,
    ).fit_transform(tsne_input)

    variant_row = results.loc[results["combo_id"].astype(str) == str(selected_variant_id)].iloc[0]
    figure, axis = plt.subplots(figsize=(9, 6.5))
    for role in ("A0", "A1", "B", "C"):
        role_mask = tsne_metadata[role_column].astype(str).eq(role).to_numpy()
        if not role_mask.any():
            continue
        axis.scatter(
            tsne_coordinates[role_mask, 0],
            tsne_coordinates[role_mask, 1],
            s=7,
            alpha=0.55,
            color=ROLE_COLORS[role],
            label=role,
            linewidths=0,
        )
    axis.set_title(
        f"t-SNE — {DISPLAY_NAME} — {variant_row['encoder_scope']} / "
        f"projector {variant_row['projector']} — "
        f"{CORPUS_LABELS.get(TSNE_CORPUS, TSNE_CORPUS)} ({points_note})"
    )
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    axis.legend(title="Accident-process role")
    axis.grid(False)
    figure.tight_layout()
    tsne_filename = f"tsne_{selected_variant_id}_{TSNE_CORPUS}.png"
    figure.savefig(FIGURES_DIR / tsne_filename, dpi=200, bbox_inches="tight")
    print("Saved:", FIGURES_DIR / tsne_filename)
    plt.show()
    plt.close(figure)
'''

CONFUSION_CODE = r'''from sklearn.metrics import confusion_matrix


def plot_simple_confusion(matrix_df, *, title: str, filename: str):
    labels = ["A0", "A1", "B", "C"]
    matrix = matrix_df.reindex(index=labels, columns=labels, fill_value=0)
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("Predicted accident-process role")
    axis.set_ylabel("True accident-process role")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / filename, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(figure)


for variant_index, variant_row in results.reset_index(drop=True).iterrows():
    variant_id = str(variant_row["combo_id"])
    variant_dir = RESULTS_DIR / "combos" / variant_id
    for corpus_id in ["btp", *TEST_CORPORA]:
        prediction_path = variant_dir / "predictions" / f"predictions_{corpus_id}.csv"
        if not prediction_path.is_file():
            print("Missing predictions:", prediction_path)
            continue
        predictions = pd.read_csv(prediction_path)
        if not {"true_macro", "pred_macro"}.issubset(predictions.columns):
            print("Missing true_macro/pred_macro:", prediction_path)
            continue
        labels = ["A0", "A1", "B", "C"]
        matrix = confusion_matrix(
            predictions["true_macro"].astype(str),
            predictions["pred_macro"].astype(str),
            labels=labels,
        )
        corpus_label = CORPUS_LABELS.get(corpus_id, corpus_id)
        title = (
            f"{DISPLAY_NAME} — {variant_row['encoder_scope']} / "
            f"projector {variant_row['projector']} — {corpus_label}"
        )
        plot_simple_confusion(
            pd.DataFrame(matrix, index=labels, columns=labels),
            title=title,
            filename=f"confusion_{variant_index + 1:02d}_{variant_id[:32]}_{corpus_id}.png",
        )
'''


def main() -> None:
    import sys

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from safer_core.notebook_bootstrap import NOTEBOOK_PATH_SETUP

    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for method_key, display_name in METHODS:
        config = (
            CONFIG_CODE.replace("__METHOD_KEY__", method_key)
            .replace("__DISPLAY_NAME__", display_name)
        )
        cells = [
            _cell(
                f"""# {display_name} contrastive fine-tuning results

Read-only viewer for the eight encoder/projector variants.
The table, confusion matrices and t-SNE use the same format as notebook 08.
""",
                "markdown",
            ),
            _cell(NOTEBOOK_PATH_SETUP, cell_id="bootstrap"),
            _cell(config, cell_id="config"),
            _cell("## 1 — Tuned contrastive results", "markdown"),
            _cell(TABLE_CODE, cell_id="table"),
            _cell(
                """## 2 — t-SNE for the best variant

Set `TSNE_VARIANT_ID` and `TSNE_CORPUS` in the configuration cell to change the figure.
`TSNE_MAX_POINTS = None` keeps all available points.
""",
                "markdown",
            ),
            _cell(TSNE_CODE, cell_id="tsne"),
            _cell(
                """## 3 — Confusion matrices for all variants

Simple confusion matrices are generated for Construction and every OOD corpus.
""",
                "markdown",
            ),
            _cell(CONFUSION_CODE, cell_id="confusion"),
            _cell(
                "Figures are saved under `RESULTS_DIR / figures_notebook`.",
                "markdown",
            ),
        ]
        output_path = NOTEBOOK_DIR / f"05_view_{method_key}_results.ipynb"
        output_path.write_text(
            json.dumps(_notebook(cells), indent=1, ensure_ascii=False),
            encoding="utf-8",
        )
        print("Wrote", output_path)


if __name__ == "__main__":
    main()
