"""Génère notebooks/08_view_supervised_macro_ft_results.ipynb.

Viewer lecture seule de la campagne de 8 variantes FT (tableau article)
+ drill-down confusions / prédictions pour une variante sélectionnée.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "08_view_supervised_macro_ft_results.ipynb"

import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from safer_core.notebook_bootstrap import NOTEBOOK_PATH_SETUP


def md(text: str) -> dict:
    src = [line + "\n" for line in text.strip().split("\n")]
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def py(text: str, cell_id: str | None = None) -> dict:
    lines = text.strip().split("\n")
    src = [ln + "\n" for ln in lines]
    cell: dict = {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": src,
    }
    if cell_id:
        cell["id"] = cell_id
    return cell


TITLE_MD = """# Résultats supervised_macro_ft — variantes article

Notebook **lecture seule** : tableau des **8 variantes** FT
(encoder scope × projector) + confusions / prédictions par combo.

> Ce notebook n'entraîne pas. Lancez d'abord le job, puis exécutez les cellules.

## Lancer la campagne (cluster)

```bash
cd ~/SAFER/text
sbatch jobs/tune_supervised_macro_ft.sh
```

Grille : `configs/tuning/supervised_macro_ft_grid.yaml`  
Sorties : `output/supervised_macro_ft/variants/`

## Contenu

1. Tableau article avec balanced accuracy uniquement
2. Drill-down d'une variante (CV, OOD)
3. t-SNE 2D d'une variante, colored by true accident-process role
4. Matrices de confusion de toutes les variantes
5. Checklist artefacts
"""

CONFIG_CODE = """# --- Paramètres ---
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from IPython.display import display

ROOT = TEXT_ROOT  # bootstrap

from safer_core.brand_style import apply_matplotlib_brand
from supervised_macro_ft.notebook_viz import load_saved_predictions

apply_matplotlib_brand()
sns.set_theme(style="whitegrid")

VARIANTS_DIR = ROOT / "output/supervised_macro_ft/variants"
# Run unique (optionnel) :
# RESULTS_DIR = ROOT / "output/supervised_macro_ft"
# ou une combo : VARIANTS_DIR / "combos" / "<combo_id>"

TEST_CORPORA = ["metallurgie", "caou", "nicollin"]
DEFAULT_CFG = ROOT / "configs/methods/supervised_macro_ft.yaml"
if DEFAULT_CFG.is_file():
    _cfg = yaml.safe_load(DEFAULT_CFG.read_text(encoding="utf-8"))
    TEST_CORPORA = list(_cfg.get("test_corpora") or TEST_CORPORA)

# Index de la variante à inspecter (0 = première ligne du tableau article)
COMBO_ROW = 0
# Variante utilisée pour le t-SNE. None = même variante que COMBO_ROW.
TSNE_COMBO_ID = None
# Corpus dont les embeddings projetés sont affichés dans le t-SNE.
TSNE_CORPUS = "metallurgie"
# None = tous les points disponibles. Mettre un entier uniquement si besoin
# d'accélérer l'exécution interactive.
TSNE_MAX_POINTS = None
TSNE_SEED = 42
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

FIGURES_DIR = VARIANTS_DIR / "figures_notebook"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("Variants :", VARIANTS_DIR)
print("Figures  :", FIGURES_DIR)
print("Corpora  :", TEST_CORPORA)
"""

CHECK_MD = """## 0 — Artefacts de la campagne

Si des fichiers manquent : `sbatch jobs/tune_supervised_macro_ft.sh`
"""

CHECK_CODE = """required = {
    "results_summary": VARIANTS_DIR / "results_summary.csv",
    "grid_summary": VARIANTS_DIR / "grid_summary.csv",
    "combos_dir": VARIANTS_DIR / "combos",
}
missing = []
print("Artefacts :")
for k, p in required.items():
    ok = p.is_file() if p.suffix else p.is_dir()
    print(f"  [{'OK' if ok else 'ABSENT'}] {k}: {p}")
    if not ok:
        missing.append(k)
if missing:
    raise FileNotFoundError(
        "Campagne incomplète — fichiers manquants : "
        + ", ".join(missing)
        + "\\nLancez : sbatch jobs/tune_supervised_macro_ft.sh"
    )
print("\\nOK — artefacts principaux présents.")
"""

TABLE_MD = """## 1 — Tableau complet des 8 variantes

Chaque ligne correspond à une architecture : encoder scope × projecteur MLP
(`Yes` / `No`). Le tableau donne uniquement la balanced accuracy : CV BTP
(`μ ± σ`), la BA de chaque corpus OOD, la moyenne OOD et le pire corpus OOD.
"""

TABLE_CODE = """results = pd.read_csv(VARIANTS_DIR / "results_summary.csv")
grid = pd.read_csv(VARIANTS_DIR / "grid_summary.csv")


def _read_metric_row(path, prediction_path=None):
    if path.is_file():
        frame = pd.read_csv(path)
        if not frame.empty:
            row = frame.iloc[0]
            return {
                metric: float(row[metric])
                for metric in ("accuracy", "macro_f1", "balanced_accuracy")
                if metric in row.index and pd.notna(row[metric])
            }
    if prediction_path is None or not prediction_path.is_file():
        return {}
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    predictions = pd.read_csv(prediction_path)
    if not {"true_macro", "pred_macro"}.issubset(predictions.columns):
        return {}
    predictions = predictions.dropna(subset=["true_macro", "pred_macro"])
    if predictions.empty:
        return {}
    y_true = predictions["true_macro"].astype(str)
    y_pred = predictions["pred_macro"].astype(str)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


grid_by_combo = (
    grid.drop_duplicates("combo_id").set_index("combo_id")
    if "combo_id" in grid.columns
    else pd.DataFrame()
)
metric_rows = []
for _, variant in results.iterrows():
    combo_id = str(variant["combo_id"])
    combo_dir = VARIANTS_DIR / "combos" / combo_id
    config_row = (
        grid_by_combo.loc[combo_id]
        if not grid_by_combo.empty and combo_id in grid_by_combo.index
        else pd.Series(dtype=object)
    )
    out = {
        "encoder_scope": variant.get("encoder_scope"),
        "projector": variant.get("projector"),
        "combo_id": combo_id,
        "cv_accuracy": config_row.get("mean_accuracy", np.nan),
        "cv_macro_f1": config_row.get("mean_macro_f1", np.nan),
        "cv_balanced_accuracy": config_row.get("mean_balanced_accuracy", np.nan),
    }
    for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
        out[f"cv_{metric}_std"] = config_row.get(f"std_{metric}", np.nan)
    train_metrics = _read_metric_row(
        combo_dir / "metrics" / "metrics_classification_btp.csv",
        combo_dir / "predictions" / "predictions_btp.csv",
    )
    for metric, value in train_metrics.items():
        out[f"btp_{metric}"] = value
    for corpus_id in TEST_CORPORA:
        metrics = _read_metric_row(
            combo_dir / "metrics" / f"metrics_classification_test_{corpus_id}.csv",
            combo_dir / "predictions" / f"predictions_{corpus_id}.csv",
        )
        for metric, value in metrics.items():
            out[f"{corpus_id}_{metric}"] = value
        summary_key = f"ba_ood_{corpus_id}"
        if (
            f"{corpus_id}_balanced_accuracy" not in out
            and summary_key in variant.index
            and pd.notna(variant[summary_key])
        ):
            out[f"{corpus_id}_balanced_accuracy"] = float(variant[summary_key])
    metric_rows.append(out)

metrics_table = pd.DataFrame(metric_rows)
summary_rows = []
metric_labels = {
    "balanced_accuracy": "BA",
}
for _, metric_row in metrics_table.iterrows():
    summary = {
        "Model": f"{metric_row['encoder_scope']} / projector {metric_row['projector']}"
    }
    for metric, label in metric_labels.items():
        cv_mean = metric_row.get(f"cv_{metric}", np.nan)
        cv_std = metric_row.get(f"cv_{metric}_std", np.nan)
        summary[f"CV {label} ± std (BTP)"] = (
            f"{cv_mean:.3f} ± {cv_std:.3f}"
            if pd.notna(cv_mean) and pd.notna(cv_std)
            else "—"
        )
        for corpus_id in TEST_CORPORA:
            summary[f"{label} {CORPUS_LABELS.get(corpus_id, corpus_id)}"] = metric_row.get(
                f"{corpus_id}_{metric}", np.nan
            )
        ood_values = [
            metric_row.get(f"{corpus_id}_{metric}", np.nan)
            for corpus_id in TEST_CORPORA
        ]
        ood_values = [float(value) for value in ood_values if pd.notna(value)]
        summary[f"{label} OOD Avg"] = (
            float(np.mean(ood_values)) if ood_values else np.nan
        )
        summary[f"{label} OOD Worst"] = (
            float(np.min(ood_values)) if ood_values else np.nan
        )
    summary_rows.append(summary)

summary_table = pd.DataFrame(summary_rows)
display(summary_table.style.format(
    {column: "{:.3f}" for column in summary_table.columns if "OOD" in column},
    na_rep="—",
))
"""

DRILL_MD = """## 2 — Drill-down d'une variante

Choisissez `COMBO_ROW` (ou `COMBO_ID`) pour sélectionner la variante du t-SNE.
"""

DRILL_CODE = """# COMBO_ID = "train_last_n_layers1_projectionmlp_sklearn_........"  # optionnel
COMBO_ID = globals().get("COMBO_ID", None)

if COMBO_ID:
    row = results.loc[results["combo_id"].astype(str) == str(COMBO_ID)].iloc[0]
else:
    row = results.iloc[int(COMBO_ROW)]

combo_id = str(row["combo_id"])
combo_dir = Path(str(row.get("combo_dir") or (VARIANTS_DIR / "combos" / combo_id)))
if not combo_dir.is_dir():
    combo_dir = VARIANTS_DIR / "combos" / combo_id

print("Variante :", row.get("encoder_scope"), "| projector =", row.get("projector"))
print("combo_id :", combo_id)
print("dir      :", combo_dir)
"""

TSNE_MD = """## 3 — t-SNE 2D d'une variante

Modifiez `TSNE_COMBO_ID` et `TSNE_CORPUS` dans la cellule de paramètres,
puis réexécutez cette cellule. Les couleurs représentent le **vrai rôle de
processus accidentel** (`pred_label` dans les métadonnées). Par défaut,
`TSNE_MAX_POINTS = None` conserve tous les points disponibles.
"""

TSNE_CODE = """from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

selected_tsne_id = TSNE_COMBO_ID or combo_id
tsne_rows = results.loc[results["combo_id"].astype(str) == str(selected_tsne_id)]
if tsne_rows.empty:
    raise KeyError(f"TSNE_COMBO_ID inconnu : {selected_tsne_id}")
tsne_row = tsne_rows.iloc[0]
tsne_combo_dir = VARIANTS_DIR / "combos" / str(selected_tsne_id)
tsne_emb_path = tsne_combo_dir / "embeddings" / f"projected_{TSNE_CORPUS}.npy"
tsne_meta_path = tsne_combo_dir / "embeddings" / f"projected_{TSNE_CORPUS}_metadata.csv"

if not tsne_emb_path.is_file() or not tsne_meta_path.is_file():
    print(
        "Embeddings projetés absents pour cette combinaison / ce corpus :",
        tsne_emb_path,
    )
else:
    tsne_embeddings = np.load(tsne_emb_path)
    tsne_meta = pd.read_csv(tsne_meta_path)
    if len(tsne_embeddings) != len(tsne_meta):
        raise ValueError("Embeddings et métadonnées t-SNE non alignés")
    role_col = "pred_label" if "pred_label" in tsne_meta.columns else "true_macro"
    if role_col not in tsne_meta.columns:
        raise KeyError("Aucune colonne de vrai rôle trouvée dans les métadonnées")

    if TSNE_MAX_POINTS is not None and len(tsne_embeddings) > int(TSNE_MAX_POINTS):
        rng = np.random.default_rng(TSNE_SEED)
        indices = np.sort(
            rng.choice(len(tsne_embeddings), size=int(TSNE_MAX_POINTS), replace=False)
        )
        tsne_embeddings = tsne_embeddings[indices]
        tsne_meta = tsne_meta.iloc[indices].reset_index(drop=True)
        sampling_note = f"échantillon de {len(tsne_meta):,} points"
    else:
        sampling_note = f"tous les {len(tsne_meta):,} points"

    pca_components = min(50, tsne_embeddings.shape[1], len(tsne_embeddings) - 1)
    tsne_input = PCA(
        n_components=pca_components,
        random_state=TSNE_SEED,
    ).fit_transform(tsne_embeddings)
    perplexity = min(30, max(5, (len(tsne_input) - 1) // 3))
    tsne_xy = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=TSNE_SEED,
    ).fit_transform(tsne_input)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for role in ("A0", "A1", "B", "C"):
        mask = tsne_meta[role_col].astype(str).eq(role).to_numpy()
        if not mask.any():
            continue
        ax.scatter(
            tsne_xy[mask, 0],
            tsne_xy[mask, 1],
            s=7,
            alpha=0.55,
            color=ROLE_COLORS[role],
            label=role,
            linewidths=0,
        )
    ax.set_title(
        f"t-SNE — {tsne_row.get('encoder_scope')} / projector {tsne_row.get('projector')} — "
        f"{CORPUS_LABELS.get(TSNE_CORPUS, TSNE_CORPUS)} ({sampling_note})"
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(title="Accident-process role")
    ax.grid(False)
    fig.tight_layout()
    tsne_filename = f"03_tsne_{str(selected_tsne_id)[:40]}_{TSNE_CORPUS}.png"
    fig.savefig(FIGURES_DIR / tsne_filename, dpi=200, bbox_inches="tight")
    print("Figure t-SNE :", FIGURES_DIR / tsne_filename)
    plt.show()
    plt.close(fig)
"""

CONF_MD = """## 4 — Matrices de confusion de toutes les variantes

Pour chaque combinaison encoder/projecteur, les matrices sont affichées pour
BTP et pour chaque corpus OOD. Les prédictions sont chargées sous
`combos/<id>/predictions/`.
"""

CONF_CODE = """from sklearn.metrics import confusion_matrix


def plot_simple_confusion(cm_df, *, title: str, filename: str):
    labels = ["A0", "A1", "B", "C"]
    matrix = cm_df.reindex(index=labels, columns=labels, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted accident-process role")
    ax.set_ylabel("True accident-process role")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


corpora = ["btp", *TEST_CORPORA]
for variant_index, variant_row in results.reset_index(drop=True).iterrows():
    variant_combo_id = str(variant_row["combo_id"])
    variant_dir = VARIANTS_DIR / "combos" / variant_combo_id
    print(
        f"\\n######## Variante {variant_index + 1}/{len(results)}: "
        f"{variant_row.get('encoder_scope')} / projector {variant_row.get('projector')} ########"
    )
    for corpus_id in corpora:
        pred_df = load_saved_predictions(variant_dir, corpus_id)
        if pred_df is None:
            print(f"(absent) predictions_{corpus_id}.csv")
            continue
        true_col = "true_macro" if "true_macro" in pred_df.columns else "pred_label"
        pred_col = "pred_macro" if "pred_macro" in pred_df.columns else None
        if pred_col is None or true_col not in pred_df.columns:
            print(f"(skip) colonnes de prédiction absentes pour {corpus_id}")
            continue
        print(f"=== Confusion — {corpus_id} (n={len(pred_df)}) ===")
        cm = confusion_matrix(
            pred_df[true_col].astype(str),
            pred_df[pred_col].astype(str),
            labels=["A0", "A1", "B", "C"],
        )
        plot_simple_confusion(
            pd.DataFrame(cm, index=["A0", "A1", "B", "C"], columns=["A0", "A1", "B", "C"]),
            title=(
                f"{variant_row.get('encoder_scope')} / "
                f"projector {variant_row.get('projector')} — "
                f"{CORPUS_LABELS.get(corpus_id, corpus_id)}"
            ),
            filename=f"04_confusion_{variant_index + 1:02d}_{variant_combo_id[:32]}_{corpus_id}.png",
        )
        plt.close("all")
"""

FOOTER_MD = """---

**Figures** : `{VARIANTS_DIR}/figures_notebook/`

Run unitaire (hors campagne) : `sbatch jobs/train_supervised_macro_ft.sh`  
Baseline sklearn Qwen : notebook **07** / **07b**.
"""


def build_notebook() -> dict:
    cells = [
        md(TITLE_MD),
        py(NOTEBOOK_PATH_SETUP, cell_id="bootstrap"),
        py(CONFIG_CODE, cell_id="config"),
        md(CHECK_MD),
        py(CHECK_CODE, cell_id="check"),
        md(TABLE_MD),
        py(TABLE_CODE, cell_id="table"),
        md(DRILL_MD),
        py(DRILL_CODE, cell_id="drill"),
        md(TSNE_MD),
        py(TSNE_CODE, cell_id="tsne"),
        md(CONF_MD),
        py(CONF_CODE, cell_id="confusion"),
        md(FOOTER_MD),
    ]
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


def main() -> None:
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"Écrit : {NB_PATH}")


if __name__ == "__main__":
    main()
