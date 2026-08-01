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
        "Régression logistique multinomiale sur embeddings **standardisés** (StandardScaler). "
        "Référence linéaire rapide, interprétable via les coefficients.",
        """\
MODEL_REGISTRY["logistic_regression"] = {
    "use_scaler": True,
    "params": {
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 2000,
        "solver": "lbfgs",
    },
}""",
    ),
    (
        "random_forest",
        "Random Forest",
        "Forêt aléatoire **sans** scaling — les arbres sont insensibles à l'échelle des features. "
        "Capture des interactions non linéaires entre dimensions d'embedding.",
        """\
MODEL_REGISTRY["random_forest"] = {
    "use_scaler": False,
    "params": {
        "n_estimators": 300,
        "max_depth": None,
        "class_weight": "balanced",
    },
}""",
    ),
    (
        "xgboost",
        "XGBoost",
        "Gradient boosting sur embeddings bruts. Si import échoue : `pip install xgboost`.",
        """\
MODEL_REGISTRY["xgboost"] = {
    "use_scaler": False,
    "params": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
    },
}""",
    ),
]

TEST_CORPUS_SPECS = [
    ("metallurgie", "Métallurgie"),
    ("caou", "Caoutchouc / chimie / plastiques"),
]

INTRO_MD = r"""
# 07 — Baseline supervisée (embedding brut Qwen)

## Objectif

Établir une **ligne de base sklearn** sur les embeddings Qwen **pré-calculés** (sans fine-tuning de l'encodeur) pour la classification macro **A0 / A1 / B / C**.

Ce notebook répond à la question : *« Que vaut un classifieur classique sur les vecteurs Qwen bruts ? »* — point de comparaison pour les méthodes contrastives (05) et le fine-tuning CE (08).

## Données

| Rôle | Fichiers | Description |
|------|----------|-------------|
| **Entraînement / CV** | `dataset/data_btp.csv` + `embeddings/Qwen3-Embedding-0.6B_btp.csv` | Corpus BTP in-domain |
| **Test OOD** | `dataset/data_<corpus>.csv` + `embeddings/Qwen3-Embedding-0.6B_<corpus>.csv` | Métallurgie, caou (registre `configs/test_corpora.yaml`) |

Chaque ligne = une **unité factuelle** (`sentence`) avec `pred_label` (macro) et `accident_id` (groupe pour le CV).

## Protocole

1. **GroupKFold** (`N_FOLDS`, groupes = `accident_id`) sur BTP — un accident ne peut pas être à la fois en train et en validation.
2. **Trois classifieurs** : Logistic Regression, Random Forest, XGBoost.
3. **Sélection** du meilleur modèle sur **balanced accuracy** moyenne en CV (`SELECTION_METRIC`).
4. **Réentraînement** sur 100 % BTP → évaluation sur chaque corpus test OOD.
5. **Synthèse cross-domain** : BA CV vs BA OOD (moyenne et pire corpus).

## Métriques affichées

- **accuracy** — exactitude globale
- **balanced_accuracy** — moyenne des rappels par classe (métrique principale, déséquilibre macro)

> `macro_f1` est calculée en interne mais **non affichée** dans ce notebook (focus article sur la BA).

## Modes d'exécution

| `RESTIMATE` | Comportement |
|-------------|--------------|
| `False` *(défaut)* | **Cache** — saute CV / test si les fichiers existent déjà |
| `True` | **Force** — réentraîne tout, même si le cache est présent |

La cellule paramètres calcule automatiquement `RUN_CV` et `RUN_TEST` par corpus.

## Sorties

```
output_test/<corpus>/supervised_baseline/
├── cv/                    # CV partagée (dossier CV_CORPUS)
│   ├── cv_per_fold.csv
│   └── cv_summary.csv
├── transfer/
│   ├── target_macro_predictions.csv
│   ├── all_models_test_metrics.csv
│   └── models/<model_key>/
├── figures/               # matrices de confusion
└── cross_domain_generalization.csv
```

**Prérequis** : embeddings exportés pour BTP et corpus test (`jobs/export_corpus_embeddings.sh` ou `scripts/export_corpus_embeddings.py`).
"""

PARAMS_MD = r"""
### Paramètres généraux

**`RESTIMATE`** — contrôle unique du réentraînement :
- `False` : recharge le disque si `cv/cv_summary.csv`, `transfer/all_models_test_metrics.csv`, etc. sont déjà là ;
- `True` : relance CV + évaluation test même si les fichiers existent.

Les hyperparamètres de **chaque classifieur** restent dans les sections dédiées (étape 2).
"""

STEP1_MD = r"""
## Étape 1 — Chargement des données

Vérifie la **couverture embeddings** (chaque unité du CSV a bien un vecteur Qwen) puis charge les matrices \(X\) (dim embedding) et \(y\) (macro entière).

Le GroupKFold utilise `accident_id` : toutes les unités d'un même accident partagent le même fold.
"""

STEP2_MD = r"""
## Étape 2 — CV GroupKFold par modèle (BTP)

Pour chaque classifieur :

- entraînement sur \(K-1\) folds, évaluation sur le fold tenu à l'écart ;
- métriques **accuracy** et **balanced_accuracy** par fold ;
- les lignes sont agrégées à l'étape 3 (μ ± σ).

**StandardScaler** : appliqué avant la régression logistique (fit sur le train du fold uniquement).
"""

STEP3_MD = r"""
## Étape 3 — Synthèse CV et sélection du modèle

Tableau **μ ± σ** par modèle. Le meilleur modèle est celui qui maximise `mean_balanced_accuracy` (paramètre `SELECTION_METRIC`).

La figure barres compare les modèles sur les métriques affichées (sans `macro_f1`).
"""

STEP4_TEMPLATE = r"""
## Étape 4 — Évaluation OOD : {display_name}

Entraîne **chaque** classifieur sur 100 % BTP, prédit sur le corpus test **{display_name}** (`{corpus_id}`).

Affichage :
- tableau des métriques test (accuracy, balanced_accuracy) ;
- matrice de confusion par modèle ;
- copie de la confusion du **meilleur modèle CV** vers `figures/confusion_test.png`.
"""

STEP5_MD = r"""
## Étape 5 — Synthèse cross-domain (article)

Compare la **BA en CV BTP** (μ ± σ) à la généralisation **hors domaine** :

\[
\mathrm{BA}_{\mathrm{OOD,avg}} = \frac{1}{M}\sum_{m=1}^{M}\mathrm{BA}_{T_m},
\qquad
\mathrm{BA}_{\mathrm{OOD,worst}} = \min_m \mathrm{BA}_{T_m}
\]

Export : `cross_domain_generalization.csv` (dossier CV partagé).
"""

ARTIFACTS_MD = r"""
## Artefacts attendus

Liste de contrôle des fichiers produits (ou rechargés en mode cache).
"""


def main() -> None:
    cells = [
        md(INTRO_MD),
        md(PARAMS_MD),
        py(
            NOTEBOOK_PATH_SETUP
            + """
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from macro_transfer.constants import MACRO_NAMES
from macro_transfer.notebook_viz import plot_fsp_confusion_heatmap
from macro_transfer.supervised_baseline import (
    aggregate_cv_metrics,
    evaluate_all_models_on_test,
    export_all_models_test_results,
    export_cv_results,
    load_supervised_datasets,
    load_cached_fold_rows_for_model,
    load_cached_cv_results,
    load_cached_all_models_test_results,
    load_cached_test_results,
    load_ood_balanced_accuracy_by_corpus,
    load_supervised_run_manifest,
    require_supervised_cache,
    run_model_group_kfold_cv,
    save_supervised_run_manifest,
    select_best_model,
    summarize_cross_domain_generalization,
    supervised_baseline_output_dir,
    supervised_ml_artifacts_exist,
)
from safer_core.test_corpus import resolve_test_corpus

# --- Général ---
METHOD_NAME = "supervised_macro_baseline"
N_FOLDS = 7
SEED = 42
SELECTION_METRIC = "balanced_accuracy"

# True  → force réentraînement (CV + test), même si cache présent
# False → recharge le cache lorsque les fichiers existent ; sinon entraîne
RESTIMATE = False

# Métriques affichées dans tableaux et graphiques (macro_f1 exclue volontairement)
DISPLAY_METRICS = ("accuracy", "balanced_accuracy")

TEST_CORPORA = ["metallurgie", "caou", "nicollin"]
CV_CORPUS = "metallurgie"

SOURCE_CFG = {
    "dataset_path": "dataset/data_btp.csv",
    "emb_csv": "embeddings/Qwen3-Embedding-0.6B_btp.csv",
    "text_col": "sentence",
    "label_col": "pred_label",
    "group_col": "accident_id",
    "pred_ok_col": "pred_ok",
}

TARGET_COL_CFG = {
    "text_col": "sentence",
    "label_col": "pred_label",
    "group_col": "accident_id",
    "pred_ok_col": "pred_ok",
}

MODEL_ORDER = ["logistic_regression", "random_forest", "xgboost"]
MODEL_REGISTRY: dict = {}


def build_cfg(corpus_id: str) -> dict:
    spec = resolve_test_corpus(corpus_id, anchor=TEXT_ROOT)
    target = {
        **TARGET_COL_CFG,
        "dataset_path": str(spec.data_csv.relative_to(TEXT_ROOT)).replace("\\\\", "/"),
        "emb_csv": str(spec.emb_csv.relative_to(TEXT_ROOT)).replace("\\\\", "/"),
    }
    return {
        "method_name": METHOD_NAME,
        "corpus": corpus_id,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "selection_metric": SELECTION_METRIC,
        "source": dict(SOURCE_CFG),
        "target": target,
    }


_cv_spec = resolve_test_corpus(CV_CORPUS, anchor=TEXT_ROOT)
CV_OUT_DIR = supervised_baseline_output_dir(CV_CORPUS, anchor=TEXT_ROOT)
CV_FIG_DIR = CV_OUT_DIR / "figures"
CV_DIR = CV_OUT_DIR / "cv"
CV_FIG_DIR.mkdir(parents=True, exist_ok=True)


def _cv_cache_ready() -> bool:
    return (CV_DIR / "cv_summary.csv").is_file() and (CV_DIR / "cv_per_fold.csv").is_file()


def _test_cache_ready(out_dir: Path) -> bool:
    transfer = Path(out_dir) / "transfer"
    return (
        (transfer / "all_models_test_metrics.csv").is_file()
        and (transfer / "target_macro_predictions.csv").is_file()
    )


RUN_CV = bool(RESTIMATE) or not _cv_cache_ready()

print("Méthode :", METHOD_NAME)
print("CV folds :", N_FOLDS, "| sélection :", SELECTION_METRIC)
print("Métriques affichées :", DISPLAY_METRICS)
print("Corpus CV / artefacts :", _cv_spec.display_name, f"({CV_CORPUS})")
print("Corpus test OOD :", TEST_CORPORA)
print("Sorties CV :", CV_OUT_DIR)
print("RESTIMATE :", RESTIMATE, ("(force réentraînement)" if RESTIMATE else "(cache si présent)"))
print("RUN_CV :", RUN_CV, "| cache CV prêt :", _cv_cache_ready())
sns.set_theme(style="whitegrid")
"""
        ),
        md(STEP1_MD),
        py(
            r"""
from safer_core.data_loading import embedding_coverage_report

for label, data_path, emb_path in [
    ("BTP", SOURCE_CFG["dataset_path"], SOURCE_CFG["emb_csv"]),
    *[
        (
            resolve_test_corpus(cid, anchor=TEXT_ROOT).display_name,
            str(resolve_test_corpus(cid, anchor=TEXT_ROOT).data_csv.relative_to(TEXT_ROOT)).replace("\\", "/"),
            str(resolve_test_corpus(cid, anchor=TEXT_ROOT).emb_csv.relative_to(TEXT_ROOT)).replace("\\", "/"),
        )
        for cid in TEST_CORPORA
    ],
]:
    rep = embedding_coverage_report(data_path, emb_path, label_col=SOURCE_CFG["label_col"])
    print(f"{label}: metadata={rep['metadata_rows']} | emb={rep['embedding_rows']} | manquants={rep['missing_embeddings']}")
    if rep["missing_embeddings"] > 0:
        print(
            f"  → python scripts/export_corpus_embeddings.py --corpus <id> --force "
            f"(BTP: --corpus btp)"
        )

cfg = build_cfg(CV_CORPUS)
DATA = load_supervised_datasets(cfg, anchor=TEXT_ROOT)

X_btp = DATA["X_btp"]
y_btp = DATA["y_btp"]
groups_btp = DATA["groups_btp"]
MACROS = DATA["macros"]

print("BTP :", X_btp.shape, "| dim embedding :", X_btp.shape[1])
print("Distribution macros BTP :")
display(pd.Series(y_btp).value_counts().rename("effectif"))

for corpus_id in TEST_CORPORA:
    spec = resolve_test_corpus(corpus_id, anchor=TEXT_ROOT)
    data_c = load_supervised_datasets(build_cfg(corpus_id), anchor=TEXT_ROOT)
    print(
        f"\n{spec.display_name} ({corpus_id})",
        f"\n  data : {spec.data_csv}",
        f"\n  emb  : {spec.emb_csv}",
        f"\n  test : {data_c['X_test'].shape}",
        sep="",
    )
"""
        ),
        md(STEP2_MD),
    ]

    for model_key, title, desc, model_cfg_block in MODEL_SECTIONS:
        cells.extend(
            [
                md(f"### {title}\n\n{desc}"),
                py(
                    f"""
# --- Hyperparamètres {title} ---
{model_cfg_block}
params = MODEL_REGISTRY[{model_key!r}]["params"]
use_scaler = MODEL_REGISTRY[{model_key!r}]["use_scaler"]
print("Hyperparamètres :", params)
print("StandardScaler avant classifieur :", use_scaler)

if RUN_CV:
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
else:
    require_supervised_cache(CV_OUT_DIR, include_bertopic=False)
    fold_rows_{model_key} = load_cached_fold_rows_for_model(CV_OUT_DIR, {model_key!r})
    print("Chargé depuis cache :", CV_DIR / "cv_per_fold.csv")
fold_df = pd.DataFrame(fold_rows_{model_key})
display(fold_df[[c for c in fold_df.columns if c in ("model", "fold", *DISPLAY_METRICS)]])
"""
                ),
            ]
        )

    cells.extend(
        [
            md(STEP3_MD),
            py(
                r"""
MODEL_KEYS = [k for k in MODEL_ORDER if k in MODEL_REGISTRY]
print("Modèles enregistrés :", MODEL_KEYS)

if RUN_CV:
    all_fold_rows = []
    for key in MODEL_KEYS:
        all_fold_rows.extend(globals()[f"fold_rows_{key}"])

    cv_per_fold = pd.DataFrame(all_fold_rows)
    cv_summary = aggregate_cv_metrics(all_fold_rows)
    export_cv_results(CV_OUT_DIR, all_fold_rows, cv_summary)
else:
    require_supervised_cache(CV_OUT_DIR, include_bertopic=False)
    all_fold_rows, cv_summary = load_cached_cv_results(CV_OUT_DIR)
    cv_per_fold = pd.DataFrame(all_fold_rows)
    print("CV rechargée depuis :", CV_DIR)

display_cols = ["model"]
for m in DISPLAY_METRICS:
    display_cols.extend([f"mean_{m}", f"std_{m}"])
display(cv_summary[display_cols])

manifest = load_supervised_run_manifest(CV_OUT_DIR) if not RUN_CV else {}
best_model = (
    str(manifest.get("best_model"))
    if not RUN_CV and manifest.get("best_model")
    else select_best_model(cv_summary, selection_metric=SELECTION_METRIC)
)
print("Meilleur modèle (", SELECTION_METRIC, ") :", best_model)

plot_df = cv_summary.melt(
    id_vars=["model"],
    value_vars=[f"mean_{m}" for m in DISPLAY_METRICS],
    var_name="metric",
    value_name="mean",
)
plot_df["metric"] = plot_df["metric"].str.replace("mean_", "")
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(data=plot_df, x="model", y="mean", hue="metric", ax=ax)
ax.set_title(f"CV GroupKFold ({N_FOLDS} folds) — μ par métrique")
ax.set_ylabel("score")
ax.set_ylim(0, 1)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
fig.savefig(CV_FIG_DIR / "cv_comparison.png", dpi=150)
plt.show()
"""
            ),
        ]
    )

    for corpus_id, display_name in TEST_CORPUS_SPECS:
        cells.extend(
            [
                md(STEP4_TEMPLATE.format(display_name=display_name, corpus_id=corpus_id)),
                py(
                    f"""
corpus_id = {corpus_id!r}
_spec = resolve_test_corpus(corpus_id, anchor=TEXT_ROOT)
OUT_DIR = supervised_baseline_output_dir(corpus_id, anchor=TEXT_ROOT)
FIG_DIR = OUT_DIR / "figures"
TRANSFER_DIR = OUT_DIR / "transfer"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Corpus test : {{_spec.display_name}} ({{corpus_id}})")
print("Sorties :", OUT_DIR)

cfg_c = build_cfg(corpus_id)
DATA_C = load_supervised_datasets(cfg_c, anchor=TEXT_ROOT)
X_test = DATA_C["X_test"]
test_meta = DATA_C["test_meta"]

print("Test :", X_test.shape)
display(test_meta[[DATA_C["target_text_col"], DATA_C["target_label_col"]]].head())

RUN_TEST = bool(RESTIMATE) or not _test_cache_ready(OUT_DIR)
print("RUN_TEST :", RUN_TEST, "| cache test prêt :", _test_cache_ready(OUT_DIR))

if RUN_TEST:
    preds_by_model, metrics_by_model = evaluate_all_models_on_test(
        MODEL_KEYS,
        MODEL_REGISTRY,
        X_btp,
        y_btp,
        X_test,
        test_meta,
        macros=MACROS,
        seed=SEED,
        text_col=DATA_C["target_text_col"],
        group_col=DATA_C["target_group_col"],
        label_col=DATA_C["target_label_col"],
        method_prefix=METHOD_NAME,
    )
    all_models_summary = export_all_models_test_results(
        OUT_DIR,
        preds_by_model,
        metrics_by_model,
        macros=MACROS,
        best_model=best_model,
    )
    save_supervised_run_manifest(
        OUT_DIR,
        best_model=best_model,
        selection_metric=SELECTION_METRIC,
        seed=SEED,
        n_folds=N_FOLDS,
        test_corpus=corpus_id,
        model_keys=MODEL_KEYS,
    )
    print("Résultats ML sauvegardés sous :", TRANSFER_DIR / "models")
else:
    require_supervised_cache(OUT_DIR, include_bertopic=False)
    try:
        preds_by_model, metrics_by_model, all_models_summary = load_cached_all_models_test_results(
            OUT_DIR, MODEL_KEYS, macros=MACROS
        )
    except FileNotFoundError:
        preds, metrics = load_cached_test_results(OUT_DIR, macros=MACROS)
        manifest = load_supervised_run_manifest(OUT_DIR)
        _best = str(manifest.get("best_model", best_model))
        preds_by_model = {{_best: preds}}
        metrics_by_model = {{_best: metrics}}
        all_models_summary = pd.DataFrame(
            [{{k: v for k, v in metrics.items() if k in DISPLAY_METRICS}}]
        )
        all_models_summary.insert(0, "model", _best)
    manifest = load_supervised_run_manifest(OUT_DIR)
    best_model_corpus = str(manifest.get("best_model", best_model))
    print("Prédictions rechargées (par modèle) :", TRANSFER_DIR / "models")

print("Meilleur modèle (CV,", SELECTION_METRIC, ") :", best_model)
summary_cols = ["model"] + [c for c in all_models_summary.columns if c in DISPLAY_METRICS]
display(all_models_summary[summary_cols])

for model_key in MODEL_KEYS:
    m = metrics_by_model.get(model_key, {{}})
    if "_confusion_matrix" not in m:
        continue
    cm = np.asarray(m["_confusion_matrix"])
    cm_df = pd.DataFrame(cm, index=MACROS, columns=MACROS)
    plot_fsp_confusion_heatmap(
        cm_df,
        fig_dir=FIG_DIR,
        title=f"Confusion test — {{corpus_id}} — {{model_key}}",
        filename=f"confusion_test_{{model_key}}.png",
    )
    print(f"\\n### {{model_key}} — balanced_accuracy = {{m.get('balanced_accuracy', float('nan')):.3f}}")
    display(cm_df)

preds = preds_by_model[best_model]
metrics = metrics_by_model[best_model]
if "_confusion_matrix" in metrics:
    plot_fsp_confusion_heatmap(
        pd.DataFrame(np.asarray(metrics["_confusion_matrix"]), index=MACROS, columns=MACROS),
        fig_dir=FIG_DIR,
        title=f"Confusion test (meilleur modèle) — {{corpus_id}}",
    )
    import shutil
    src = FIG_DIR / "confusion_heatmap.png"
    if src.is_file():
        shutil.copy(src, FIG_DIR / "confusion_test.png")
"""
                ),
            ]
        )

    cells.extend(
        [
            md(STEP5_MD),
            py(
                r"""
ood_ba_by_corpus = load_ood_balanced_accuracy_by_corpus(
    TEST_CORPORA, MODEL_KEYS, anchor=TEXT_ROOT
)
cross_domain_summary = summarize_cross_domain_generalization(
    cv_summary, ood_ba_by_corpus, model_keys=MODEL_KEYS
)
display(
    cross_domain_summary.rename(
        columns={
            "model": "model",
            "cv_ba": "CV ± std (BTP)",
            "ba_ood_avg": "BA OOD Avg",
            "ba_ood_worst": "BA OOD Worst",
        }
    )[["model", "CV ± std (BTP)", "BA OOD Avg", "BA OOD Worst"]]
)
cross_domain_summary.to_csv(
    CV_OUT_DIR / "cross_domain_generalization.csv", index=False
)
print("Export :", CV_OUT_DIR / "cross_domain_generalization.csv")
"""
            ),
            md(ARTIFACTS_MD),
            py(
                r"""
print("Artefacts :")
expected = [
    CV_DIR / "cv_summary.csv",
    CV_FIG_DIR / "cv_comparison.png",
    CV_OUT_DIR / "cross_domain_generalization.csv",
]
for corpus_id in TEST_CORPORA:
    out_dir = supervised_baseline_output_dir(corpus_id, anchor=TEXT_ROOT)
    expected.extend([
        out_dir / "transfer" / "target_macro_predictions.csv",
        out_dir / "transfer" / "all_models_test_metrics.csv",
        out_dir / "transfer" / "metrics.json",
        out_dir / "figures" / "confusion_test.png",
    ])
for p in expected:
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
