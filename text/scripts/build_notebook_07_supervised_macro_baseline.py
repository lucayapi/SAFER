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
3. Réentraînement 100 % BTP → évaluation **métallurgie** (tous les modèles + meilleur pour BERTopic)
4. BERTopic intra-macro sur `pred_macro` (config alignée FSP) — **DataMapPlot en cellule séparée** (fin du notebook)

**`RESTIMATE_ML=True`** (défaut) : réentraîne CV + classifieurs + évaluation test.  
**`RESTIMATE_BERTOPIC=True`** (défaut) : relance BERTopic intra-macro (+ judge LLM si activé).  
Mettre l’un ou l’autre à **`False`** pour recharger les artefacts déjà produits sous `OUT_DIR`.

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
    show_bertopic_datamaps_inline,
)
from macro_transfer.report_tables import load_macro_topic_stats
from macro_transfer.supervised_baseline import (
    DEFAULT_MODEL_REGISTRY,
    aggregate_cv_metrics,
    evaluate_all_models_on_test,
    export_all_models_test_results,
    export_cv_results,
    load_supervised_config,
    load_supervised_datasets,
    load_cached_fold_rows_for_model,
    load_cached_cv_results,
    load_cached_all_models_test_results,
    load_cached_test_results,
    load_supervised_run_manifest,
    merge_model_registry,
    require_supervised_cache,
    run_all_models_group_kfold_cv,
    run_model_group_kfold_cv,
    run_supervised_bertopic_phase,
    save_supervised_run_manifest,
    select_best_model,
    supervised_baseline_output_dir,
    supervised_ml_artifacts_exist,
)
from safer_core.test_corpus import resolve_test_corpus

# --- Parameters (modifier ici ou via papermill) ---
TEST_CORPUS = "metallurgie"
N_FOLDS = 5
SEED = 42
SELECTION_METRIC = "macro_f1"
RESTIMATE_ML = True        # True = CV + réentraînement + test | False = cache ML
RESTIMATE_BERTOPIC = True  # True = relancer BERTopic (+ judge) | False = cache topics
CONFIG_PATH = TEXT_ROOT / "configs" / "supervised_macro_baseline.yaml"
_BASE_CFG = load_supervised_config(CONFIG_PATH)
BERTOPIC_CFG = dict(_BASE_CFG.get("bertopic") or {})
TOPICS_EXPORT_CFG = dict(_BASE_CFG.get("topics_export") or {})
TOPIC_JUDGE_CFG = dict(_BASE_CFG.get("topic_judge") or {})

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
print("RESTIMATE_ML :", RESTIMATE_ML, ("(réentraînement)" if RESTIMATE_ML else "(cache disque)"))
print("RESTIMATE_BERTOPIC :", RESTIMATE_BERTOPIC, ("(réentraînement)" if RESTIMATE_BERTOPIC else "(cache disque)"))
if not RESTIMATE_ML:
    print("Cache ML OK :", supervised_ml_artifacts_exist(OUT_DIR))
print("Modèles :", MODEL_KEYS)
print("BERTopic (HDBSCAN) :", BERTOPIC_CFG.get("hdbscan", {}))
print("BERTopic (UMAP) :", BERTOPIC_CFG.get("umap", {}))
print("BERTopic min_topic_size :", BERTOPIC_CFG.get("min_topic_size"))
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

if RESTIMATE_ML:
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
    require_supervised_cache(OUT_DIR, include_bertopic=False)
    fold_rows_{model_key} = load_cached_fold_rows_for_model(OUT_DIR, {model_key!r})
    print("Chargé depuis cache :", CV_DIR / "cv_per_fold.csv")
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
if RESTIMATE_ML:
    all_fold_rows = []
    for key in MODEL_KEYS:
        all_fold_rows.extend(globals()[f"fold_rows_{key}"])

    cv_per_fold = pd.DataFrame(all_fold_rows)
    cv_summary = aggregate_cv_metrics(all_fold_rows)
    export_cv_results(OUT_DIR, all_fold_rows, cv_summary)
else:
    require_supervised_cache(OUT_DIR, include_bertopic=False)
    all_fold_rows, cv_summary = load_cached_cv_results(OUT_DIR)
    cv_per_fold = pd.DataFrame(all_fold_rows)
    print("CV rechargée depuis :", CV_DIR)

display_cols = ["model"]
for m in ("accuracy", "macro_f1", "balanced_accuracy"):
    display_cols.extend([f"mean_{m}", f"std_{m}"])
display(cv_summary[display_cols])

manifest = load_supervised_run_manifest(OUT_DIR) if not RESTIMATE_ML else {}
best_model = (
    str(manifest.get("best_model"))
    if not RESTIMATE_ML and manifest.get("best_model")
    else select_best_model(cv_summary, selection_metric=SELECTION_METRIC)
)
print("Meilleur modèle (", SELECTION_METRIC, ") :", best_model)

# Barres comparatives (recalcul ou depuis cache)
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
            md("## Étape 4 — Tous les modèles sur métallurgie"),
            py(
                r"""
from macro_transfer.supervised_baseline import (
    evaluate_all_models_on_test,
    export_all_models_test_results,
    load_cached_all_models_test_results,
    load_cached_test_results,
    save_supervised_run_manifest,
)

if RESTIMATE_ML:
    preds_by_model, metrics_by_model = evaluate_all_models_on_test(
        MODEL_KEYS,
        MODEL_REGISTRY,
        X_btp,
        y_btp,
        X_test,
        test_meta,
        macros=MACROS,
        seed=SEED,
        text_col=DATA["target_text_col"],
        group_col=DATA["target_group_col"],
        label_col=DATA["target_label_col"],
        method_prefix="supervised_macro_baseline",
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
        test_corpus=TEST_CORPUS,
        model_keys=MODEL_KEYS,
    )
    print("Résultats ML sauvegardés sous :", OUT_DIR / "transfer" / "models")
else:
    require_supervised_cache(OUT_DIR, include_bertopic=False)
    try:
        preds_by_model, metrics_by_model, all_models_summary = load_cached_all_models_test_results(
            OUT_DIR, MODEL_KEYS, macros=MACROS
        )
    except FileNotFoundError:
        preds, metrics = load_cached_test_results(OUT_DIR, macros=MACROS)
        manifest = load_supervised_run_manifest(OUT_DIR)
        best_model = str(manifest.get("best_model", best_model))
        preds_by_model = {best_model: preds}
        metrics_by_model = {best_model: metrics}
        all_models_summary = pd.DataFrame(
            [{k: v for k, v in metrics.items() if k in ("accuracy", "macro_f1", "balanced_accuracy")}]
        )
        all_models_summary.insert(0, "model", best_model)
    manifest = load_supervised_run_manifest(OUT_DIR)
    best_model = str(manifest.get("best_model", best_model))
    print("Prédictions rechargées (par modèle) :", OUT_DIR / "transfer" / "models")

print("Meilleur modèle (CV,", SELECTION_METRIC, ") :", best_model)
display(all_models_summary)

# Matrices de confusion — un panneau par modèle
for model_key in MODEL_KEYS:
    m = metrics_by_model.get(model_key, {})
    if "_confusion_matrix" not in m:
        continue
    cm = np.asarray(m["_confusion_matrix"])
    cm_df = pd.DataFrame(cm, index=MACROS, columns=MACROS)
    plot_fsp_confusion_heatmap(
        cm_df,
        fig_dir=FIG_DIR,
        title=f"Confusion test — {model_key}",
        filename=f"confusion_test_{model_key}.png",
    )
    print(f"\n### {model_key}")
    display(cm_df)
    cls_rep = m.get("_classification_report")
    if cls_rep:
        display(pd.DataFrame(cls_rep).T)

preds = preds_by_model[best_model]
metrics = metrics_by_model[best_model]
if "_confusion_matrix" in metrics:
    plot_fsp_confusion_heatmap(
        pd.DataFrame(np.asarray(metrics["_confusion_matrix"]), index=MACROS, columns=MACROS),
        fig_dir=FIG_DIR,
        title="Confusion test (meilleur modèle)",
    )
    import shutil
    src = FIG_DIR / "confusion_heatmap.png"
    if src.is_file():
        shutil.copy(src, FIG_DIR / "confusion_test.png")
"""
            ),
            md("## Étape 5 — BERTopic intra-macro"),
            py(
                r"""
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

print("Paramètres BERTopic :")
display(pd.DataFrame([{
    "min_topic_size": BERTOPIC_CFG.get("min_topic_size"),
    "macro_params": "eom / min_cluster=8 (config partagée FSP ↔ supervisé)",
    "include_ambiguous": BERTOPIC_CFG.get("include_ambiguous", False),
}]))
print(json.dumps(BERTOPIC_CFG, indent=2, ensure_ascii=False))

bertopic_cfg = dict(BERTOPIC_CFG)
bertopic_cfg["topic_judge"] = dict(TOPIC_JUDGE_CFG)
_diag = dict(bertopic_cfg.get("diagnostics") or {})
_diag["save_datamap"] = False
bertopic_cfg["diagnostics"] = _diag
topics_export_cfg = dict(TOPICS_EXPORT_CFG)

if RESTIMATE_BERTOPIC:
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
    print("BERTopic terminé :", bertopic_summary.get("topics_dir", OUT_DIR / "topics_bertopic"))
else:
    require_supervised_cache(OUT_DIR, include_bertopic=True)
    bertopic_summary = {"topics_dir": str(OUT_DIR / "topics_bertopic"), "cached": True}
    print("BERTopic rechargé depuis :", OUT_DIR / "topics_bertopic")
"""
            ),
            md("### Évaluation LLM-judge des topics"),
            md(
                r"""
Chaque topic est noté sur 6 critères (0–5) ; le **score_global** est calculé en Python.
Verdict : conserver / fusionner / scinder / rejeter. Sorties : `summary/topic_judge_*.csv`.
"""
            ),
            py(
                r"""
from macro_transfer.notebook_viz import (
    load_topic_judge_artifacts,
    plot_topic_judge_quality,
)
from macro_transfer.topic_judge import run_topic_judge_evaluation

_judge_scores_path = OUT_DIR / "summary" / "topic_judge_scores.csv"
# Le judge est déjà lancé par le pipeline BERTopic si RESTIMATE_BERTOPIC=True.
# Ici : compléter uniquement si scores absents (ex. run BERTopic antérieur sans judge).
if TOPIC_JUDGE_CFG.get("enabled", False) and not _judge_scores_path.is_file():
    themes_path = OUT_DIR / "topics_bertopic" / "themes_by_macro.csv"
    assign_path = OUT_DIR / "topics_bertopic" / "assignments.csv"
    if themes_path.is_file() and assign_path.is_file():
        run_topic_judge_evaluation(
            OUT_DIR,
            test_meta,
            pd.read_csv(assign_path),
            pd.read_csv(themes_path),
            cfg=dict(TOPIC_JUDGE_CFG),
            text_col=DATA["target_text_col"],
            seed=SEED,
            force=False,
        )
    else:
        print("BERTopic absent — judge ignoré.")

_judge_art = load_topic_judge_artifacts(OUT_DIR)
if _judge_art["scores"].empty:
    print("topic_judge_scores.csv absent.")
else:
    display(_judge_art["scores"].head(20))
    if not _judge_art["macro_summary"].empty:
        display(_judge_art["macro_summary"])
plot_topic_judge_quality(OUT_DIR, fig_dir=FIG_DIR, show=True)
"""
            ),
            md("### Tableau topics par étape (Rôle × unités / bruit)"),
            py(
                r"""
df_stats = load_macro_topic_stats(OUT_DIR)
if df_stats.empty:
    print("macro_topic_stats.csv absent — relancer la cellule BERTopic ci-dessus.")
else:
    table_macro = pd.DataFrame(
        {
            "Rôle": df_stats["macro"].astype(str),
            "Unités": pd.to_numeric(df_stats["n_units"], errors="coerce").fillna(0).astype(int),
            "Topics": pd.to_numeric(df_stats["n_topics"], errors="coerce").fillna(0).astype(int),
            "Bruit": pd.to_numeric(df_stats["bruit_pct"], errors="coerce").map(
                lambda x: f"{x:.1f}%" if pd.notna(x) else "--"
            ),
            "Plus gros topic": pd.to_numeric(df_stats["plus_gros_topic_pct"], errors="coerce").map(
                lambda x: f"{x:.1f}%" if pd.notna(x) else "--"
            ),
        }
    )
    display(table_macro)
    latex_macro = table_macro.to_latex(index=False, escape=False, column_format="lcccc")
    print(latex_macro)
    tex_out = OUT_DIR / "summary" / "macro_topic_stats_table.tex"
    tex_out.parent.mkdir(parents=True, exist_ok=True)
    tex_out.write_text(latex_macro, encoding="utf-8")
    print("LaTeX écrit :", tex_out)
"""
            ),
            md("### Détail thèmes BERTopic"),
            py(
                r"""
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
"""
            ),
            md("### Explorer un topic — toutes les phrases"),
            py(
                r"""
# --- Modifier ici ---
TOPIC_MACRO = "A0"   # A0 | A1 | B | C
TOPIC_NUM = 0        # topic_id BERTopic (entier >= 0)

assign_path = OUT_DIR / "topics_bertopic" / "assignments.csv"
text_col = DATA["target_text_col"]
themes_path = OUT_DIR / "topics_bertopic" / "themes_by_macro.csv"

if not assign_path.is_file():
    print("assignments.csv absent — exécuter la cellule BERTopic ci-dessus.")
else:
    assign = pd.read_csv(assign_path)
    macro_s = str(TOPIC_MACRO).strip()
    tid = int(TOPIC_NUM)
    sub = assign[
        (assign["macro"].astype(str) == macro_s)
        & (pd.to_numeric(assign["topic_id"], errors="coerce").fillna(-1).astype(int) == tid)
    ].copy()
    if sub.empty:
        known = sorted(
            pd.to_numeric(
                assign.loc[assign["macro"].astype(str) == macro_s, "topic_id"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
        )
        print(f"Aucune phrase pour macro={macro_s!r} topic_id={tid}.")
        print("Topics disponibles pour cette macro :", known)
    else:
        meta_idx = test_meta.reset_index(drop=True)
        doc_ids = sub["doc_idx"].astype(int).to_numpy()
        rows = []
        for j, doc_idx in enumerate(doc_ids):
            row = {
                "n": j + 1,
                "doc_idx": int(doc_idx),
                "sentence": str(meta_idx.iloc[int(doc_idx)][text_col]),
            }
            if "prob" in sub.columns:
                row["prob_topic"] = float(sub.iloc[j]["prob"])
            if "p_mk" in sub.columns:
                row["p_mk"] = float(sub.iloc[j]["p_mk"])
            rows.append(row)
        topic_df = pd.DataFrame(rows)
        if themes_path.is_file():
            themes = pd.read_csv(themes_path)
            th_row = themes[
                (themes["macro"].astype(str) == macro_s)
                & (pd.to_numeric(themes["topic_id"], errors="coerce").astype(int) == tid)
            ]
            if len(th_row):
                print("Libellé :", th_row.iloc[0].get("theme_label", ""))
                print("Mots-clés :", th_row.iloc[0].get("top_words", ""))
        print(f"Macro {macro_s} · topic {tid} · {len(topic_df)} phrase(s)")
        display(topic_df[["n", "doc_idx", "sentence"] + [c for c in ("prob_topic", "p_mk") if c in topic_df.columns]])
        print("\n--- Texte intégral ---")
        for _, r in topic_df.iterrows():
            print(f"[{int(r['n'])}] {r['sentence']}\n")
"""
            ),
            md(
                r"""
## DataMapPlot BERTopic (optionnel — cellule indépendante)

Génère les cartes **après** BERTopic, à partir des modèles sauvegardés sous `bertopic/<macro>/bertopic_model.pkl`.
Peut prendre **plusieurs minutes** sur A0/A1 (milliers de points). Ne nécessite pas `RESTIMATE_BERTOPIC=True` si les modèles existent déjà.

`EXPORT_DATAMAPS=True` pour régénérer ; `False` pour afficher uniquement les PNG déjà produits.
"""
            ),
            py(
                r"""
from macro_transfer.bertopic_exports import export_bertopic_datamaps_from_run

EXPORT_DATAMAPS = True  # False = afficher les PNG existants sans recalcul

if EXPORT_DATAMAPS:
    _datamap_paths = export_bertopic_datamaps_from_run(
        OUT_DIR,
        test_meta,
        X_test,
        macros=MACROS,
        text_col=DATA["target_text_col"],
        fig_dir=FIG_DIR,
        show_progress=True,
    )
    print("DataMapPlot exportés :", _datamap_paths)
else:
    print("EXPORT_DATAMAPS=False — affichage des fichiers existants uniquement.")

datamap_paths = show_bertopic_datamaps_inline(OUT_DIR, macros=MACROS)
if not datamap_paths:
    print(
        "Aucune carte — exécuter avec EXPORT_DATAMAPS=True après BERTopic "
        "(modèles attendus : bertopic/<macro>/bertopic_model.pkl)."
    )
"""
            ),
            py(
                r"""
print("Artefacts :")
expected = [
    CV_DIR / "cv_summary.csv",
    TRANSFER_DIR / "target_macro_predictions.csv",
    TRANSFER_DIR / "all_models_test_metrics.csv",
    TRANSFER_DIR / "metrics.json",
    OUT_DIR / "topics_bertopic" / "themes_by_macro.csv",
    FIG_DIR / "cv_comparison.png",
    FIG_DIR / "confusion_test.png",
]
for macro in MACRO_NAMES:
    expected.append(FIG_DIR / f"bertopic_datamap_{macro}.png")
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
