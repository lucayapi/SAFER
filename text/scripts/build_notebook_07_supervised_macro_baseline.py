"""Génère notebooks/07_supervised_macro_baseline.ipynb (baseline supervisée Qwen brut)."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "07_supervised_macro_baseline.ipynb"

import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from safer_core.notebook_bootstrap import NOTEBOOK_PATH_SETUP


def md(text: str) -> dict:
    src = [line + "\n" for line in text.strip().split("\n")]
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def py(text: str) -> dict:
    lines = text.strip().split("\n")
    src = [ln + "\n" for ln in lines]
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": src,
    }


MODEL_SECTIONS = [
    (
        "logistic_regression",
        "Logistic Regression",
        "StandardScaler + LogisticRegression — `C=1.0`, `class_weight='balanced'`, `max_iter=2000`, `solver='lbfgs'`.",
    ),
    (
        "random_forest",
        "Random Forest",
        "Pas de scaling — `n_estimators=300`, `max_depth=None`, `class_weight='balanced'`.",
    ),
    (
        "xgboost",
        "XGBoost",
        "Pas de scaling — `n_estimators=300`, `max_depth=6`, `learning_rate=0.1`, `objective='multi:softprob'`. "
        "Si import échoue : `pip install xgboost`.",
    ),
    (
        "mlp",
        "MLP",
        "StandardScaler + MLPClassifier — `hidden_layer_sizes=(256,128)`, `max_iter=500`, `early_stopping=True`.",
    ),
]


def main() -> None:
    cells = [
        md(
            r"""
# 07 — Baseline supervisée (embedding brut Qwen)

Pipeline exécutable sur **embeddings Qwen bruts** (BTP) :
1. **GroupKFold** (5 folds, `accident_id`) — LR, Random Forest, XGBoost, MLP
2. Synthèse CV (μ ± σ) et sélection du meilleur modèle (`macro_f1`)
3. Réentraînement 100 % BTP → évaluation **métallurgie**
4. BERTopic intra-macro sur `pred_macro` (config alignée FSP)

**Prérequis** : embeddings BTP et test exportés (`export_raw_geometry.sh`, `export_test_embeddings.sh`).
"""
        ),
        py(
            NOTEBOOK_PATH_SETUP
            + """
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from macro_transfer.constants import MACRO_NAMES
from macro_transfer.notebook_viz import (
    build_topics_display_dataframe,
    plot_fsp_confusion_heatmap,
)
from macro_transfer.supervised_baseline import (
    DEFAULT_MODEL_REGISTRY,
    aggregate_cv_metrics,
    export_cv_results,
    export_test_results,
    fit_final_and_predict_test,
    load_supervised_config,
    load_supervised_datasets,
    merge_model_registry,
    run_all_models_group_kfold_cv,
    run_model_group_kfold_cv,
    run_supervised_bertopic_phase,
    select_best_model,
    supervised_baseline_output_dir,
)
from safer_core.test_corpus import resolve_test_corpus

# --- Parameters (modifier ici ou via papermill) ---
TEST_CORPUS = "metallurgie"
N_FOLDS = 5
SEED = 42
SELECTION_METRIC = "macro_f1"
CONFIG_PATH = TEXT_ROOT / "configs" / "supervised_macro_baseline.yaml"

MODEL_REGISTRY = merge_model_registry({
    "logistic_regression": DEFAULT_MODEL_REGISTRY["logistic_regression"],
    "random_forest": DEFAULT_MODEL_REGISTRY["random_forest"],
    "xgboost": DEFAULT_MODEL_REGISTRY["xgboost"],
    "mlp": DEFAULT_MODEL_REGISTRY["mlp"],
})
MODEL_KEYS = list(MODEL_REGISTRY.keys())

_spec = resolve_test_corpus(TEST_CORPUS, anchor=TEXT_ROOT)
OUT_DIR = supervised_baseline_output_dir(_spec.id, anchor=TEXT_ROOT)
FIG_DIR = OUT_DIR / "figures"
CV_DIR = OUT_DIR / "cv"
TRANSFER_DIR = OUT_DIR / "transfer"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Corpus test : {_spec.display_name} ({_spec.id})")
print("Sorties :", OUT_DIR)
print("Modèles :", MODEL_KEYS)
sns.set_theme(style="whitegrid")
"""
        ),
        md("## Étape 1 — Chargement BTP + test (Qwen brut)"),
        py(
            r"""
cfg = load_supervised_config(CONFIG_PATH)
cfg = dict(cfg)
cfg["corpus"] = TEST_CORPUS
DATA = load_supervised_datasets(cfg, anchor=TEXT_ROOT)

X_btp = DATA["X_btp"]
y_btp = DATA["y_btp"]
groups_btp = DATA["groups_btp"]
X_test = DATA["X_test"]
test_meta = DATA["test_meta"]
MACROS = DATA["macros"]

print("BTP :", X_btp.shape, "| macros :", pd.Series(y_btp).value_counts().to_dict())
print("Test :", X_test.shape)
display(test_meta[[DATA["target_text_col"], DATA["target_label_col"]]].head())
"""
        ),
        md("## Étape 2 — CV GroupKFold par modèle"),
    ]

    for model_key, title, desc in MODEL_SECTIONS:
        cells.extend(
            [
                md(f"### {title}\n\n{desc}"),
                py(
                    f"""
params = MODEL_REGISTRY[{model_key!r}].get("params", {{}})
use_scaler = MODEL_REGISTRY[{model_key!r}].get("use_scaler")
print("Hyperparamètres :", params)
print("Scaler :", use_scaler)

fold_rows_{model_key} = run_model_group_kfold_cv(
    {model_key!r},
    X_btp,
    y_btp,
    groups_btp,
    macros=MACROS,
    n_folds=N_FOLDS,
    seed=SEED,
    params=params,
    use_scaler=use_scaler,
)
display(pd.DataFrame(fold_rows_{model_key}))
"""
                ),
            ]
        )

    cells.extend(
        [
            md("## Étape 3 — Synthèse CV (μ ± σ)"),
            py(
                r"""
all_fold_rows = []
for key in MODEL_KEYS:
    all_fold_rows.extend(globals()[f"fold_rows_{key}"])

cv_per_fold = pd.DataFrame(all_fold_rows)
cv_summary = aggregate_cv_metrics(all_fold_rows)
export_cv_results(OUT_DIR, all_fold_rows, cv_summary)

display_cols = ["model"]
for m in ("accuracy", "macro_f1", "balanced_accuracy"):
    display_cols.extend([f"mean_{m}", f"std_{m}"])
display(cv_summary[display_cols])

best_model = select_best_model(cv_summary, selection_metric=SELECTION_METRIC)
print("Meilleur modèle (", SELECTION_METRIC, ") :", best_model)

# Barres comparatives
plot_df = cv_summary.melt(
    id_vars=["model"],
    value_vars=[f"mean_{m}" for m in ("accuracy", "macro_f1", "balanced_accuracy")],
    var_name="metric",
    value_name="mean",
)
plot_df["metric"] = plot_df["metric"].str.replace("mean_", "")
fig, ax = plt.subplots(figsize=(9, 4))
sns.barplot(data=plot_df, x="model", y="mean", hue="metric", ax=ax)
ax.set_title(f"CV GroupKFold ({N_FOLDS} folds) — μ par métrique")
ax.set_ylabel("score")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
fig.savefig(FIG_DIR / "cv_comparison.png", dpi=150)
plt.show()
"""
            ),
            md("## Étape 4 — Meilleur modèle sur métallurgie"),
            py(
                r"""
best_spec = MODEL_REGISTRY[best_model]
_, preds, metrics = fit_final_and_predict_test(
    best_model,
    X_btp,
    y_btp,
    X_test,
    test_meta,
    macros=MACROS,
    seed=SEED,
    params=best_spec.get("params"),
    use_scaler=best_spec.get("use_scaler"),
    method_name=f"supervised_macro_baseline/{best_model}",
    text_col=DATA["target_text_col"],
    group_col=DATA["target_group_col"],
    label_col=DATA["target_label_col"],
)
export_test_results(OUT_DIR, preds, metrics, macros=MACROS)

print("Métriques test :")
display(pd.DataFrame([{k: v for k, v in metrics.items() if not str(k).startswith("_")}]))

if "_confusion_matrix" in metrics:
    cm = np.asarray(metrics["_confusion_matrix"])
    cm_df = pd.DataFrame(cm, index=MACROS, columns=MACROS)
    plot_fsp_confusion_heatmap(cm_df, fig_dir=FIG_DIR, title="Confusion test (métallurgie)")
    # Copie avec le nom attendu par le plan
    src = FIG_DIR / "confusion_heatmap.png"
    if src.is_file():
        import shutil
        shutil.copy(src, FIG_DIR / "confusion_test.png")
    display(cm_df)

cls_rep = metrics.get("_classification_report")
if cls_rep:
    display(pd.DataFrame(cls_rep).T)
"""
            ),
            md("## Étape 5 — BERTopic intra-macro"),
            py(
                r"""
bertopic_cfg = dict(cfg.get("bertopic") or {})
topics_export_cfg = dict(cfg.get("topics_export") or {})

bertopic_summary = run_supervised_bertopic_phase(
    OUT_DIR,
    test_meta=test_meta,
    preds=preds,
    X_test=X_test,
    macros=MACROS,
    bertopic_cfg=bertopic_cfg,
    topics_export_cfg=topics_export_cfg,
    text_col=DATA["target_text_col"],
    corpus_id=TEST_CORPUS,
    method_name=f"supervised_macro_baseline/{best_model}",
    anchor=TEXT_ROOT,
)
print("BERTopic :", bertopic_summary.get("topics_dir", OUT_DIR / "topics_bertopic"))

themes_path = OUT_DIR / "topics_bertopic" / "themes_by_macro.csv"
if themes_path.is_file():
    themes = pd.read_csv(themes_path)
    display(themes.head(20))
    try:
        meta_for_topics = test_meta.copy()
        meta_for_topics["m_hat"] = preds["pred_macro"].astype(str).to_numpy()
        topics_display = build_topics_display_dataframe(OUT_DIR, meta_for_topics)
        display(topics_display.head(15))
    except FileNotFoundError as exc:
        print("Affichage topics détaillé indisponible :", exc)
else:
    print("themes_by_macro.csv absent (BERTopic désactivé ou échec).")

print("Artefacts :")
for p in [
    CV_DIR / "cv_summary.csv",
    TRANSFER_DIR / "target_macro_predictions.csv",
    TRANSFER_DIR / "metrics.json",
    themes_path,
    FIG_DIR / "cv_comparison.png",
    FIG_DIR / "confusion_test.png",
]:
    print(" ", p, "→", "OK" if p.is_file() else "absent")
"""
            ),
        ]
    )

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("OK:", NB_PATH)


if __name__ == "__main__":
    main()
