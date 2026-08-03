"""Génère notebooks/07b_supervised_macro_baseline_tuned.ipynb (lecture résultats du job)."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "07b_supervised_macro_baseline_tuned.ipynb"

import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from safer_core.notebook_bootstrap import NOTEBOOK_PATH_SETUP


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


def py(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


INTRO = r"""
# 07b — Résultats du tuning baseline supervisée

Notebook **lecture seule** : affiche les sorties du job Slurm de grille HP
(LR / Random Forest / XGBoost) sur embeddings Qwen bruts.

> Ce notebook **n'entraîne pas**. Lance d'abord le job, puis exécute les cellules.

## Lancer le job (cluster)

```bash
cd ~/SAFER/text
sbatch jobs/tune_supervised_macro_baseline.sh
```

Grille : `configs/tuning/supervised_macro_baseline_grid.yaml`

## Contenu

1. Central tuned results table with BA CV / OOD
2. **t-SNE** séparés des embeddings gelés, with a shared accident-process palette
3. **Confusion matrices** train (BTP) + OOD (all models / corpora)
4. Checklist des artefacts

## Chemins attendus

```
output[_test]/<corpus>/supervised_baseline/tuning/
  results_summary.csv, grid_summary.csv, best_*.json

output[_test]/btp/supervised_baseline_tuned/transfer/
  source_macro_predictions.csv, models/<model>/…

output[_test]/<corpus>/supervised_baseline_tuned/transfer/
  target_macro_predictions.csv, all_models_test_metrics.csv, models/<model>/…
```
"""


def main() -> None:
    cells = [
        md(INTRO),
        md("### Paramètres"),
        py(
            NOTEBOOK_PATH_SETUP
            + """
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

from macro_transfer.constants import MACRO_NAMES
from macro_transfer.supervised_baseline import (
    load_cached_all_models_test_results,
    load_supervised_run_manifest,
    supervised_baseline_output_dir,
)
from macro_transfer.supervised_baseline_tuning import (
    compare_default_vs_tuned_cv,
    load_tuning_artifacts,
    supervised_baseline_tuned_output_dir,
    supervised_baseline_tuning_dir,
)
from safer_core.brand_style import apply_matplotlib_brand
from safer_core.test_corpus import resolve_test_corpus

SELECTION_METRIC = "balanced_accuracy"
DISPLAY_METRICS = ("accuracy", "balanced_accuracy")
TEST_CORPORA = ["metallurgie", "caou", "nicollin"]
CV_CORPUS = "metallurgie"
MODEL_ORDER = ["logistic_regression", "random_forest", "xgboost"]
MACROS = list(MACRO_NAMES)
ROLE_COLORS = {
    "A0": "#0072B2",
    "A1": "#E69F00",
    "B": "#009E73",
    "C": "#D55E00",
}


def plot_simple_confusion(cm_df, *, title: str, fig_dir: Path, filename: str):
    labels = list(MACROS)
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
    fig.savefig(fig_dir / filename, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

OUTPUT_ROOTS = [TEXT_ROOT / "output", TEXT_ROOT / "output_test"]


def _resolve_layout_dir(corpus_id: str, method: str) -> Path:
    candidates = [root / str(corpus_id) / method for root in OUTPUT_ROOTS]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _resolve_tuning_dir(corpus_id: str) -> Path:
    candidates = [
        root / str(corpus_id) / "supervised_baseline" / "tuning"
        for root in OUTPUT_ROOTS
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


DEFAULT_DIRS = {
    corpus_id: _resolve_layout_dir(corpus_id, "supervised_baseline")
    for corpus_id in TEST_CORPORA
}
TUNED_DIRS = {
    corpus_id: _resolve_layout_dir(corpus_id, "supervised_baseline_tuned")
    for corpus_id in TEST_CORPORA + ["btp"]
}
TUNING_DIRS = {corpus_id: _resolve_tuning_dir(corpus_id) for corpus_id in TEST_CORPORA}
TUNING_DIR = TUNING_DIRS[CV_CORPUS]
DEFAULT_CV_DIR = DEFAULT_DIRS[CV_CORPUS]
TUNED_CV_DIR = TUNED_DIRS[CV_CORPUS]
TRAIN_OUT_DIR = TUNED_DIRS["btp"]
FIG_DIR = TUNED_CV_DIR / "figures_notebook"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Racines de sorties :", OUTPUT_ROOTS)
print("Tuning par corpus :", TUNING_DIRS)
print("Baseline 07 par corpus :", DEFAULT_DIRS)
print("Tuned 07b par corpus :", TUNED_DIRS)
print("Figures    :", FIG_DIR)
apply_matplotlib_brand()
sns.set_theme(style="whitegrid")
"""
        ),
        md(
            """
## 0 — Vérification des artefacts du job

Si des fichiers manquent, relance / attends la fin de :
`sbatch jobs/tune_supervised_macro_baseline.sh`
"""
        ),
        py(
            r"""
required = {
    "train_preds_best": TRAIN_OUT_DIR / "transfer" / "source_macro_predictions.csv",
}
for corpus_id in TEST_CORPORA:
    out = TUNED_DIRS[corpus_id]
    required[f"ood_metrics_{corpus_id}"] = out / "transfer" / "all_models_test_metrics.csv"
    required[f"ood_preds_{corpus_id}"] = out / "transfer" / "target_macro_predictions.csv"
    tuning_results = TUNING_DIRS[corpus_id] / "results_summary.csv"
    if tuning_results.is_file():
        required[f"results_summary_{corpus_id}"] = tuning_results

if not any((path / "grid_summary.csv").is_file() for path in TUNING_DIRS.values()):
    required["grid_summary"] = TUNING_DIR / "grid_summary.csv"

missing = [k for k, p in required.items() if not p.is_file()]
print("Artefacts :")
for k, p in required.items():
    print(f"  [{'OK' if p.is_file() else 'ABSENT'}] {k}: {p}")

if missing:
    raise FileNotFoundError(
        "Job de tuning incomplet — fichiers manquants : "
        + ", ".join(missing)
        + "\\nLancez : sbatch jobs/tune_supervised_macro_baseline.sh"
    )
print("\\nTous les artefacts principaux sont présents.")
"""
        ),
        md(
            """
## 1 — Tuned baseline results

Central tuned results with balanced accuracy only.
"""
        ),
        py(
            r"""
results_summary = pd.read_csv(TUNING_DIR / "results_summary.csv")
if "model" in results_summary.columns:
    cat = pd.Categorical(results_summary["model"], categories=MODEL_ORDER, ordered=True)
    results_summary = results_summary.assign(model=cat).sort_values("model")

display_columns = {
    "model": "Model",
    "cv_balanced_accuracy": "CV BA (BTP)",
    "cv_ba_std": "CV BA std",
    "ba_ood_metallurgie": "BA Metallurgy",
    "ba_ood_caou": "BA Chemistry-plastics",
    "ba_ood_nicollin": "BA Company",
    "ba_ood_avg": "BA OOD Avg",
    "ba_ood_worst": "BA OOD Worst",
}
show_cols = [c for c in display_columns if c in results_summary.columns]
central_results = results_summary[show_cols].rename(columns=display_columns).copy()
numeric_cols = [c for c in central_results.columns if c != "Model"]
display(central_results.style.format("{:.3f}", subset=numeric_cols, na_rep="—"))
"""
        ),
        md("## 2 — Top combos de la grille (toutes les tentatives)"),
        py(
            r"""
ordered = grid_summary.sort_values("selection_score", ascending=False, kind="mergesort")
show_cols = [
    c
    for c in (
        "model",
        "combo_id",
        "selection_score",
        "mean_balanced_accuracy",
        "std_balanced_accuracy",
        "mean_accuracy",
        "best_params",
    )
    if c in ordered.columns
]
print(f"{len(ordered)} combos au total")
display(ordered[show_cols].head(25))

# Best score par modèle (boxplot / strip de la grille)
if {"model", "selection_score"}.issubset(grid_summary.columns):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=grid_summary, x="model", y="selection_score", order=MODEL_ORDER, ax=ax)
    sns.stripplot(
        data=grid_summary,
        x="model",
        y="selection_score",
        order=MODEL_ORDER,
        color="0.3",
        size=3,
        ax=ax,
    )
    ax.set_ylabel("CV balanced accuracy")
    ax.set_title("Distribution des scores de grille par modèle")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "grid_score_distribution.png", dpi=150)
    plt.show()

fig, ax = plt.subplots(figsize=(11, 4))
top = ordered.head(15).copy()
top["label"] = (
    top["model"].astype(str) + "\\n" + top["combo_id"].astype(str).str.slice(0, 32)
)
sns.barplot(data=top, x="label", y="selection_score", hue="model", dodge=False, ax=ax)
ax.set_ylabel("CV balanced accuracy")
ax.set_xlabel("")
ax.set_title("Top 15 hyperparameter combos")
plt.xticks(rotation=30, ha="right", fontsize=7)
plt.tight_layout()
fig.savefig(FIG_DIR / "grid_top_combos.png", dpi=150)
plt.show()
"""
        ),
        md(
            """
## 3 — Défaut (notebook 07) vs tuné (job 07b) — CV BTP

Compare `cv_summary` du baseline non tuné si disponible.
"""
        ),
        py(
            r"""
cv_summary_tuned = pd.read_csv(TUNED_CV_DIR / "cv" / "cv_summary.csv")
default_cv_path = DEFAULT_CV_DIR / "cv" / "cv_summary.csv"
if default_cv_path.is_file():
    cv_summary_default = pd.read_csv(default_cv_path)
    print("Baseline 07 chargée :", default_cv_path)
else:
    cv_summary_default = pd.DataFrame(
        columns=["model", "mean_balanced_accuracy", "std_balanced_accuracy"]
    )
    print("Baseline 07 absente — comparaison CV partielle.")

cmp = compare_default_vs_tuned_cv(
    cv_summary_default, cv_summary_tuned, metric=SELECTION_METRIC
)
cmp_disp = cmp.copy()
for col in ("cv_ba_default", "cv_ba_tuned", "delta_ba"):
    if col in cmp_disp.columns:
        cmp_disp[col] = cmp_disp[col].map(
            lambda x: f"{x:.3f}" if pd.notna(x) else "—"
        )
display(
    cmp_disp.rename(
        columns={
            "model": "Model",
            "cv_ba_default": "CV BA default (07)",
            "cv_ba_tuned": "CV BA tuned (07b)",
            "delta_ba": "Δ BA",
        }
    )[["Model", "CV BA default (07)", "CV BA tuned (07b)", "Δ BA"]]
)

plot_cmp = cmp.dropna(subset=["cv_ba_default", "cv_ba_tuned"], how="any")
if not plot_cmp.empty:
    melt = plot_cmp.melt(
        id_vars=["model"],
        value_vars=["cv_ba_default", "cv_ba_tuned"],
        var_name="setting",
        value_name="ba",
    )
    melt["setting"] = melt["setting"].map(
        {"cv_ba_default": "Default (07)", "cv_ba_tuned": "Tuned (07b)"}
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=melt, x="model", y="ba", hue="setting", ax=ax)
    ax.set_ylabel("CV balanced accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Default vs tuned hyperparameters (BTP CV)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "default_vs_tuned_cv.png", dpi=150)
    plt.show()
"""
        ),
        md("## 4 — Synthèse cross-domain (params tunés)"),
        py(
            r"""
cross_parts = []
for corpus_id in TEST_CORPORA:
    candidates = [
        TUNED_DIRS[corpus_id] / "cross_domain_generalization.csv",
        TUNING_DIRS[corpus_id] / "cross_domain_generalization.csv",
    ]
    cross_path = next((path for path in candidates if path.is_file()), None)
    if cross_path is not None:
        part = pd.read_csv(cross_path)
        part.insert(0, "corpus", corpus_id)
        cross_parts.append(part)
cross_tuned = pd.concat(cross_parts, ignore_index=True) if cross_parts else pd.DataFrame()
if cross_tuned.empty:
    print("Aucune synthèse cross-domain trouvée.")
cross_disp = cross_tuned.rename(
    columns={
        "model": "Model",
        "cv_ba": "CV ± std (BTP)",
        "ba_ood_avg": "BA OOD Avg",
        "ba_ood_worst": "BA OOD Worst",
    }
)
keep = [c for c in ("Model", "CV ± std (BTP)", "BA OOD Avg", "BA OOD Worst") if c in cross_disp.columns]
display(cross_disp[keep])

# Métriques OOD détaillées par corpus
ood_tables = []
for corpus_id in TEST_CORPORA:
    path = (
        TUNED_DIRS[corpus_id]
        / "transfer"
        / "all_models_test_metrics.csv"
    )
    df = pd.read_csv(path)
    df.insert(0, "corpus", resolve_test_corpus(corpus_id, anchor=TEXT_ROOT).display_name)
    ood_tables.append(df)
ood_all = pd.concat(ood_tables, ignore_index=True)
disp_cols = ["corpus", "model"] + [c for c in DISPLAY_METRICS if c in ood_all.columns]
display(ood_all[disp_cols].sort_values(["corpus", "model"]))

# Default vs tuned OOD (si 07 dispo)
rows = []
for corpus_id in TEST_CORPORA:
    def_path = (
        DEFAULT_DIRS[corpus_id]
        / "transfer"
        / "all_models_test_metrics.csv"
    )
    tun_path = (
        TUNED_DIRS[corpus_id]
        / "transfer"
        / "all_models_test_metrics.csv"
    )
    if not (def_path.is_file() and tun_path.is_file()):
        continue
    d = pd.read_csv(def_path).set_index("model")
    t = pd.read_csv(tun_path).set_index("model")
    for model in sorted(set(d.index.astype(str)) & set(t.index.astype(str))):
        rows.append(
            {
                "corpus": resolve_test_corpus(corpus_id, anchor=TEXT_ROOT).display_name,
                "model": model,
                "ba_default": float(d.loc[model, "balanced_accuracy"]),
                "ba_tuned": float(t.loc[model, "balanced_accuracy"]),
                "delta": float(t.loc[model, "balanced_accuracy"])
                - float(d.loc[model, "balanced_accuracy"]),
            }
        )

ood_cmp = pd.DataFrame(rows)
if ood_cmp.empty:
    print("Comparaison OOD 07 vs 07b incomplete (lancez le notebook 07 si besoin).")
else:
    display(ood_cmp.sort_values(["corpus", "model"]))
    fig, ax = plt.subplots(figsize=(11, 4))
    melt = ood_cmp.melt(
        id_vars=["corpus", "model"],
        value_vars=["ba_default", "ba_tuned"],
        var_name="setting",
        value_name="ba",
    )
    melt["setting"] = melt["setting"].map(
        {"ba_default": "Default (07)", "ba_tuned": "Tuned (07b)"}
    )
    melt["x"] = melt["corpus"].astype(str) + " | " + melt["model"].astype(str)
    sns.barplot(data=melt, x="x", y="ba", hue="setting", ax=ax)
    ax.set_ylabel("OOD balanced accuracy")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.set_title("Default vs tuned — OOD test corpora")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "default_vs_tuned_ood.png", dpi=150)
    plt.show()
"""
        ),
        md(
            """
## 2 — Frozen embeddings t-SNE

Each corpus is shown in a separate figure. Colors represent the **true
accident-process role** (`pred_label`). All available points are projected;
no sampling is applied.
"""
        ),
        py(
            r"""
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from safer_core.data_loading import load_metadata_with_embeddings

TSNE_SEED = 42
ROLE_COLORS = {
    "A0": "#4C78A8",
    "A1": "#F58518",
    "B": "#54A24B",
    "C": "#E45756",
}
def _load_frozen_embeddings(corpus_id: str):
    spec = resolve_test_corpus(corpus_id, anchor=TEXT_ROOT, require_files=True)
    meta, dim_cols = load_metadata_with_embeddings(
        spec.data_csv,
        spec.emb_csv,
        label_col="pred_label",
        pred_ok_col="pred_ok",
        group_col="accident_id",
    )
    role = meta["pred_label"].astype(str)
    keep = role.isin(MACROS).to_numpy()
    meta = meta.loc[keep].reset_index(drop=True)
    embeddings = meta[dim_cols].to_numpy(dtype=np.float32, copy=True)
    return meta, embeddings


def _plot_frozen_tsne(corpus_id: str, title: str, filename: str):
    plot_meta, plot_embeddings = _load_frozen_embeddings(corpus_id)
    pca_components = min(50, plot_embeddings.shape[1], len(plot_embeddings) - 1)
    reduced_embeddings = PCA(
        n_components=pca_components,
        random_state=TSNE_SEED,
    ).fit_transform(plot_embeddings)
    perplexity = min(30, max(5, (len(plot_embeddings) - 1) // 3))
    projection = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=TSNE_SEED,
    ).fit_transform(reduced_embeddings)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for role in MACROS:
        mask = plot_meta["pred_label"].eq(role).to_numpy()
        if not mask.any():
            continue
        ax.scatter(
            projection[mask, 0],
            projection[mask, 1],
            s=7,
            alpha=0.55,
            c=ROLE_COLORS[role],
            marker="o",
            linewidths=0,
            label=role,
        )

    role_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor=ROLE_COLORS[role],
            markersize=8,
            label=role,
        )
        for role in MACROS
    ]
    ax.legend(handles=role_handles, title="Accident-process role")
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(False)
    fig.tight_layout()
    output_path = FIG_DIR / filename
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved figure: {output_path} ({len(plot_meta):,} points, no sampling)")
    plt.show()
    plt.close(fig)


_plot_frozen_tsne(
    "btp",
    "Frozen embeddings - Construction",
    "tsne_frozen_construction.png",
)
_plot_frozen_tsne(
    "metallurgie",
    "Frozen embeddings - Metallurgy",
    "tsne_frozen_metallurgie.png",
)
_plot_frozen_tsne(
    "caou",
    "Frozen embeddings - Chemistry-plastics",
    "tsne_frozen_chimie_plastiques.png",
)
_plot_frozen_tsne(
    "nicollin",
    "Frozen embeddings - Company",
    "tsne_frozen_company.png",
)
"""
        ),
        md(
            """
## 3 — Confusion matrices — train BTP

Predictions : `output_test/btp/supervised_baseline_tuned/transfer/models/<model>/`
"""
        ),
        py(
            r"""
def _load_source_cm(model_key: str):
    model_dir = TRAIN_OUT_DIR / "transfer" / "models" / model_key
    cm_path = model_dir / "source_confusion_matrix.csv"
    metrics_path = model_dir / "source_metrics.json"
    preds_path = model_dir / "source_macro_predictions.csv"
    if cm_path.is_file():
        return pd.read_csv(cm_path, index_col=0), metrics_path
    if preds_path.is_file():
        preds = pd.read_csv(preds_path)
        if {"true_macro", "pred_macro"}.issubset(preds.columns):
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(
                preds["true_macro"].astype(str),
                preds["pred_macro"].astype(str),
                labels=MACROS,
            )
            return pd.DataFrame(cm, index=MACROS, columns=MACROS), metrics_path
    return None, metrics_path

print("Train BTP confusion matrices")
for model_key in MODEL_ORDER:
    cm_df, metrics_path = _load_source_cm(model_key)
    if cm_df is None:
        print(f"[skip] train confusion absente pour {model_key}")
        continue
    ba = None
    if metrics_path.is_file():
        meta = json.loads(metrics_path.read_text(encoding="utf-8"))
        ba = meta.get("balanced_accuracy")
    title = f"Train BTP (tuned) — {model_key}"
    if ba is not None:
        title += f" | BA={float(ba):.3f}"
    print(f"\\n### {title}")
    plot_simple_confusion(
        cm_df,
        fig_dir=FIG_DIR,
        title=title,
        filename=f"confusion_train_{model_key}.png",
    )
    display(cm_df)
"""
        ),
        md(
            """
## 4 — Confusion matrices — OOD (all models × corpora)

Pour chaque corpus test : tableau de métriques + heatmap par modèle.
"""
        ),
        py(
            r"""
for corpus_id in TEST_CORPORA:
    out_dir = TUNED_DIRS[corpus_id]
    fig_dir = out_dir / "figures_notebook"
    fig_dir.mkdir(parents=True, exist_ok=True)
    spec = resolve_test_corpus(corpus_id, anchor=TEXT_ROOT)
    try:
        preds_by_model, metrics_by_model, summary = load_cached_all_models_test_results(
            out_dir, MODEL_ORDER, macros=MACROS
        )
    except FileNotFoundError as exc:
        print(f"{corpus_id}: cache tuné incomplet ({exc})")
        continue

    print(f"\\n{'=' * 60}")
    print(f"=== {spec.display_name} ({corpus_id}) ===")
    print(f"{'=' * 60}")
    cols = ["model"] + [c for c in DISPLAY_METRICS if c in summary.columns]
    display(summary[cols])

    for model_key in MODEL_ORDER:
        m = metrics_by_model.get(model_key, {})
        if "_confusion_matrix" not in m:
            # fallback fichier
            cm_path = out_dir / "transfer" / "models" / model_key / "confusion_matrix.csv"
            if not cm_path.is_file():
                print(f"  [skip] pas de confusion pour {model_key}")
                continue
            cm_df = pd.read_csv(cm_path, index_col=0)
        else:
            cm_df = pd.DataFrame(
                np.asarray(m["_confusion_matrix"]), index=MACROS, columns=MACROS
            )
        ba = m.get("balanced_accuracy", float("nan"))
        corpus_label = {
            "metallurgie": "Metallurgy",
            "caou": "Chemistry-plastics",
            "nicollin": "Company",
        }.get(corpus_id, corpus_id)
        title = (
            f"{corpus_label} OOD — {model_key} | BA={float(ba):.3f}"
            if ba == ba
            else f"{corpus_label} OOD — {model_key}"
        )
        print(f"\\n### {title}")
        plot_simple_confusion(
            cm_df,
            fig_dir=FIG_DIR,
            title=title,
            filename=f"confusion_ood_{corpus_id}_{model_key}.png",
        )
        # Copie sous le dossier corpus
        src = FIG_DIR / f"confusion_ood_{corpus_id}_{model_key}.png"
        dst = fig_dir / src.name
        if src.is_file() and src.resolve() != dst.resolve():
            import shutil

            shutil.copy(src, dst)
        display(cm_df)
"""
        ),
        md("## 5 — Checklist artefacts"),
        py(
            r"""
expected = [
    TUNING_DIR / "grid_summary.csv",
    TUNING_DIR / "results_summary.csv",
    TUNING_DIR / "best_combo.json",
    TUNING_DIR / "best_hyperparams.json",
    TUNED_CV_DIR / "cv" / "cv_summary.csv",
    TUNED_CV_DIR / "cross_domain_generalization.csv",
    TRAIN_OUT_DIR / "transfer" / "source_macro_predictions.csv",
]
for model_key in MODEL_ORDER:
    expected.append(
        TRAIN_OUT_DIR / "transfer" / "models" / model_key / "source_macro_predictions.csv"
    )
for corpus_id in TEST_CORPORA:
    out_dir = TUNED_DIRS[corpus_id]
    expected.append(out_dir / "transfer" / "all_models_test_metrics.csv")
    expected.append(out_dir / "transfer" / "target_macro_predictions.csv")
    for model_key in MODEL_ORDER:
        expected.append(
            out_dir / "transfer" / "models" / model_key / "target_macro_predictions.csv"
        )

print("Artefacts :")
n_ok = 0
for p in expected:
    ok = p.is_file()
    n_ok += int(ok)
    print(" ", "OK    " if ok else "ABSENT", p)
print(f"\\n{n_ok}/{len(expected)} fichiers présents")
print("Figures notebook :", FIG_DIR)
"""
        ),
    ]

    removed_sections = (
        "## 2 — Top combos",
        "## 3 — Défaut",
        "## 4 — Synthèse cross-domain",
    )
    filtered_cells = []
    skip_section = False
    for cell in cells:
        source = "".join(cell.get("source", []))
        is_heading = cell.get("cell_type") == "markdown" and source.lstrip().startswith("## ")
        if is_heading and any(source.lstrip().startswith(prefix) for prefix in removed_sections):
            skip_section = True
            continue
        if skip_section and is_heading:
            skip_section = False
        if not skip_section:
            filtered_cells.append(cell)
    cells = filtered_cells

    # Build the notebook after removing non-central tuning sections.
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"OK: {NB_PATH}")


if __name__ == "__main__":
    main()
