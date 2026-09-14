"""Génère le notebook 07c TF-IDF (paramètres modifiables dans les cellules)."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from safer_core.notebook_bootstrap import NOTEBOOK_PATH_SETUP

NB_PATH = TEXT_ROOT / "notebooks" / "07c_supervised_macro_tfidf_baseline.ipynb"


def code(source: str, tag: str):
    return nbf.v4.new_code_cell(source.strip(), metadata={"tags": [tag]})


def build_notebook():
    md = nbf.v4.new_markdown_cell
    cells = [
        md("""# 07c — Baseline supervisée lexicale : TF-IDF

## Objectif
Comparer **Logistic Regression, Random Forest et XGBoost**, comme dans le notebook 07,
avec des **unigrammes + bigrammes TF-IDF** calculés sur les textes.
Les fichiers d'embeddings Qwen ne sont pas utilisés.

**Protocole :** CV groupée par accident sur BTP → sélection sur la balanced accuracy
moyenne → entraînement final sur tout BTP → évaluation métallurgie, caou et Nicollin.
Le vocabulaire et l'IDF sont appris sur le train uniquement, y compris dans chaque fold.

**Hypothèses :** labels `A0 / A1 / B / C`, filtre `pred_ok` et labels valides identique
au projet. Les scores mesurent l'accord avec ces annotations. Les corpus cibles ne
servent ni à sélectionner le modèle ni à ajuster les paramètres.

Les tableaux affichent **accuracy, balanced accuracy et macro-F1**, complétés par
précision/rappel/F1 par classe et matrices de confusion. Le ± de la CV est la
dispersion entre folds, pas un intervalle de confiance ni une variabilité multi-seeds.

Pour chaque corpus cible, le notebook ajoute un **IC bootstrap à 95 %** pour la
balanced accuracy et le macro-F1. Il rééchantillonne des accidents complets
(`accident_id`) ; les trois modèles utilisent les mêmes tirages. Cet IC mesure
l'incertitude liée au corpus cible, pas la variabilité entre seeds.

Exécuter les cellules dans l'ordre. Dépendances : environnement Python du projet,
dont scikit-learn, XGBoost, pandas, matplotlib et seaborn. Aucun GPU ni accès API requis.
"""),
        md("""## Paramètres généraux
Modifier les corpus, colonnes, nombre de folds et seed ici. Par défaut, **7 folds**
comme le notebook 07 actuel. `RESTIMATE=False` réutilise seulement les caches
complets et compatibles ; `True` force les calculs.
Après modification d'un paramètre ou du fichier de stopwords, **relancer toutes les cellules**.
"""),
        code(NOTEBOOK_PATH_SETUP + '''
import pandas as pd
from IPython.display import Markdown, display

from macro_transfer.tfidf_baseline import (
    MACROS, METRICS, MODEL_NAMES, TfidfExperiment,
    plot_cv_summary, show_target_results,
)
from macro_transfer.target_bootstrap import (
    bootstrap_target_predictions,
    plot_target_bootstrap_intervals,
)
from safer_core.test_corpus import resolve_test_corpus

N_FOLDS = 7
SEED = 42
N_JOBS = 4  # parallélisme CPU des forêts et de XGBoost
SELECTION_METRIC = "balanced_accuracy"
RESTIMATE = False
TEST_CORPORA = ["metallurgie", "caou", "nicollin"]
OUTPUT_DIR = TEXT_ROOT / "output" / "tfidf_baseline"

# IC OOD : rééchantillonnage d'accidents entiers sur chaque corpus cible.
# Ce n'est pas une répétition multi-seeds : les trois modèles sont entraînés
# une fois avec SEED et les IC décrivent l'incertitude d'échantillonnage cible.
BOOTSTRAP_N_RESAMPLES = 2000
BOOTSTRAP_SEED = 2027
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_METRICS = ("balanced_accuracy", "macro_f1")

SOURCE_CFG = {
    "dataset_path": str(TEXT_ROOT / "dataset" / "data_btp.csv"),
    "text_col": "sentence",
    "label_col": "pred_label",
    "group_col": "accident_id",
    "pred_ok_col": "pred_ok",
}
TARGET_COL_CFG = {
    "text_col": "sentence", "label_col": "pred_label",
    "group_col": "accident_id", "pred_ok_col": "pred_ok",
}
''', "parameters"),
        md("""### Paramètres TF-IDF et stopwords

| Paramètre | Effet |
|---|---|
| `ngram_range=(1, 2)` | Mots seuls et paires ; `(2, 2)` pour les bigrammes uniquement |
| `min_df=2` | Un terme doit apparaître dans au moins 2 textes du train |
| `max_df=1.0` | Aucun retrait supplémentaire selon la fréquence maximale |
| `max_features=50000` | Plafond de taille du vocabulaire ; `None` pour aucun plafond |
| `sublinear_tf=True` | Réduit le poids des répétitions avec une fréquence logarithmique |
| `norm="l2"` | Normalise chaque vecteur ; aucun StandardScaler n'est ajouté |
| `lowercase=True`, `strip_accents=None` | Minuscules, accents conservés |

`USE_STOPWORDS` active la liste métier ; `STOPWORDS_FILE` permet de choisir une autre
liste. Une entrée par ligne, commentaires commençant par `#`. La liste est normalisée
avec la même tokenisation que les textes. Les bigrammes sont formés **après retrait
des stopwords**, entre les mots restants. Aucune autre liste n'est ajoutée automatiquement.

Référence : [TfidfVectorizer — scikit-learn](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).
"""),
        code(r'''
TFIDF_PARAMS = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 1.0,
    "max_features": 50000,
    "sublinear_tf": True,
    "norm": "l2",
    "lowercase": True,
    "strip_accents": None,
    "token_pattern": r"(?u)\b\w\w+\b",
}
USE_STOPWORDS = True
STOPWORDS_FILE = TEXT_ROOT / "macro_transfer" / "stop_metier.txt"
MODEL_REGISTRY = {}
''', "tfidf_parameters"),
        md("### Hyperparamètres — Logistic Regression\nRéférence linéaire, pondérée selon les classes du train, sur la matrice TF-IDF creuse."),
        code('''
MODEL_REGISTRY["logistic_regression"] = {
    "C": 1.0, "class_weight": "balanced",
    "max_iter": 2000, "solver": "lbfgs",
}
''', "model_logistic_regression"),
        md("### Hyperparamètres — Random Forest\nMême famille de modèle et mêmes réglages initiaux que le notebook 07."),
        code('''
MODEL_REGISTRY["random_forest"] = {
    "n_estimators": 300, "max_depth": None, "class_weight": "balanced",
}
''', "model_random_forest"),
        md("""### Hyperparamètres — XGBoost
Classification multiclasse, sans pondération supplémentaire comme dans le notebook 07.
Le calcul utilise la matrice creuse et `tree_method="hist"` sur CPU.
XGBoost applique sa gestion des valeurs manquantes aux entrées absentes de la matrice
creuse : son traitement des zéros diffère de celui des deux modèles sklearn.
Référence : [introduction Python XGBoost](https://xgboost.readthedocs.io/en/release_2.1.0/python/python_intro.html).
"""),
        code('''
MODEL_REGISTRY["xgboost"] = {
    "n_estimators": 300, "max_depth": 6, "learning_rate": 0.1,
    "objective": "multi:softprob", "eval_metric": "mlogloss",
}
''', "model_xgboost"),
        md("""## Étape 1 — Chargement des textes et contrôle des données
Les exclusions ci-dessous proviennent du filtre `pred_ok` / labels valides.
Les textes manquants deviennent des chaînes vides ; ils restent dans les scores.
Les unités dont le vecteur est nul après TF-IDF seront comptées et identifiées dans les prédictions.
"""),
        code('''
TARGET_SPECS = {
    corpus: {
        **TARGET_COL_CFG,
        "dataset_path": str(resolve_test_corpus(corpus, anchor=TEXT_ROOT).data_csv),
    }
    for corpus in TEST_CORPORA
}
CONFIG = {
    "output_dir": str(OUTPUT_DIR), "source": SOURCE_CFG, "targets": TARGET_SPECS,
    "n_folds": N_FOLDS, "seed": SEED, "n_jobs": N_JOBS,
    "selection_metric": SELECTION_METRIC, "models": MODEL_REGISTRY,
    "tfidf": TFIDF_PARAMS, "use_stopwords": USE_STOPWORDS,
    "stopwords_file": str(STOPWORDS_FILE),
}
experiment = TfidfExperiment(CONFIG, restimate=RESTIMATE)
print("Stopwords :", STOPWORDS_FILE if USE_STOPWORDS else "désactivés")
print("Nombre de termes normalisés :", len(experiment.stopwords))
print("Sorties :", OUTPUT_DIR)
display(experiment.data_summary)
for corpus, metadata in experiment.metadata.items():
    display(Markdown(f"**Aperçu — {corpus}**"))
    spec = experiment.specs[corpus]
    display(metadata[[spec["text_col"], spec["label_col"], spec["group_col"]]].head(3))
''', "load_data"),
        md("""## Étape 2 — CV GroupKFold par modèle (BTP)
Les partitions sont communes aux trois modèles. Chaque fold apprend son propre
vocabulaire et son IDF sur le train, puis transforme la validation sans réajustement.
Les scores portent sur toutes les unités de validation, y compris les vecteurs nuls.
"""),
    ]
    for model, name in [("logistic_regression", "Logistic Regression"),
                        ("random_forest", "Random Forest"), ("xgboost", "XGBoost")]:
        cells.extend([
            md(f"### {name} — validation croisée"),
            code(f'''fold_scores = experiment.run_cv({model!r})
display(fold_scores[["fold_id", "n_train", "n_val", "n_features", "n_zero_train", "n_zero_val", *METRICS]])''', f"cv_{model}"),
        ])
    cells.extend([
        md("""## Étape 3 — Synthèse CV et sélection
Tableau et barres : moyenne ± écart-type **entre folds**, sur une seule seed.
La sélection utilise exclusivement la métrique choisie en CV BTP.
"""),
        code('''
cv_summary = experiment.summarize_cv()
cv_display = pd.DataFrame({"Modèle": cv_summary["model"].map(MODEL_NAMES)})
for metric in METRICS:
    cv_display[metric] = [
        f"{mean:.3f} ± {std:.3f}"
        for mean, std in zip(cv_summary[f"mean_{metric}"], cv_summary[f"std_{metric}"])
    ]
display(cv_display)
print("Meilleur modèle CV BTP :", MODEL_NAMES[experiment.best_model])
plot_cv_summary(cv_summary, OUTPUT_DIR / "figures")
''', "cv_summary"),
        md("""## Étape 4 — Statistiques après TF-IDF
Le vocabulaire est appris sur l'intégralité du BTP, puis fixé. Le tableau décrit
les matrices TF-IDF obtenues pour BTP et pour chaque corpus cible avec ce même
vocabulaire : taille, nombre de coefficients non nuls (`nnz`), densité, nombre
moyen de caractéristiques actives par document et vecteurs nuls.

`vocabulary_size_btp` est identique sur toutes les lignes par construction. Une
forte part de vecteurs nuls dans un corpus cible indique que son lexique est peu
couvert par le vocabulaire BTP après le filtrage et `min_df`. Les corpus cibles
ne modifient jamais le vocabulaire ni l'IDF.
"""),
        code('''
tfidf_statistics = experiment.describe_tfidf_corpora()
display(tfidf_statistics.style.format({
    "density_pct": "{:.4f}%", "mean_nonzero_features_per_document": "{:.1f}",
    "median_nonzero_features_per_document": "{:.1f}", "zero_tfidf_pct": "{:.2f}%",
}))
print("Statistiques sauvegardées :", OUTPUT_DIR / "tfidf_corpus_statistics.csv")
''', "tfidf_statistics"),
        md("""## Étape 5 — Entraînement final sur tout le BTP
Un vocabulaire final commun est appris sur BTP. Chaque classifieur est entraîné
une seule fois, puis réutilisé pour les trois corpus cibles. Les modèles et le
vectoriseur sont sauvegardés avec joblib pour permettre leur réutilisation.
"""),
        code('''final_diagnostics = experiment.fit_final()
display(pd.DataFrame([final_diagnostics]))''', "final_fit"),
        md("""## Étape 6 — Évaluation des corpus cibles
Chaque corpus reçoit son tableau de comparaison et, pour chaque modèle, son rapport
par classe et deux matrices de confusion. Une ligne sans support dans une matrice
normalisée est affichée à zéro. Le macro-F1 est toujours calculé sur les quatre classes.

Le tableau suivant ajoute les IC bootstrap percentile à 95 % pour balanced accuracy
et macro-F1. Les 2 000 échantillons par défaut sont composés d'`accident_id` tirés
avec remise, en conservant toutes leurs unités factuelles ensemble.
"""),
        code('''
for corpus in TEST_CORPORA:
    display(Markdown(f"## {resolve_test_corpus(corpus, anchor=TEXT_ROOT).display_name}"))
    target_summary, target_results = experiment.evaluate_target(corpus)
    show_target_results(
        corpus, target_summary, target_results, experiment.best_model,
        OUTPUT_DIR / "targets" / corpus / "figures",
    )
    target_predictions = {
        model: pd.read_csv(record["directory"] / "predictions.csv")
        for model, record in target_results.items()
    }
    target_root = OUTPUT_DIR / "targets" / corpus
    bootstrap_intervals = bootstrap_target_predictions(
        target_predictions,
        destination=target_root / "bootstrap_ci.csv",
        n_resamples=BOOTSTRAP_N_RESAMPLES,
        seed=BOOTSTRAP_SEED,
        confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
        metrics=BOOTSTRAP_METRICS,
        force=RESTIMATE,
    )
    bootstrap_display = bootstrap_intervals[[
        "model", "metric", "point_estimate", "ci_low", "ci_high",
        "n_accidents", "n_units", "n_resamples",
    ]].copy()
    bootstrap_display["IC à 95 %"] = [
        f"[{low:.3f} ; {high:.3f}]"
        for low, high in zip(bootstrap_display.pop("ci_low"), bootstrap_display.pop("ci_high"))
    ]
    display(Markdown("#### IC bootstrap à 95 % — rééchantillonnage par accident_id"))
    display(bootstrap_display.style.format({"point_estimate": "{:.3f}"}))
    plot_target_bootstrap_intervals(
        bootstrap_intervals,
        destination=target_root / "figures" / "bootstrap_intervals.png",
        title=f"{resolve_test_corpus(corpus, anchor=TEXT_ROOT).display_name} — IC bootstrap à 95 % par accident",
    )
    print("IC exportés :", target_root / "bootstrap_ci.csv")
''', "target_evaluation"),
        md("""## Étape 7 — Synthèse cross-domain
La moyenne OOD donne le même poids à chaque corpus ; le pire score est le minimum
sur les corpus sélectionnés. La colonne `selected_on_btp_cv` repère le modèle choisi
avant évaluation cible. Les colonnes détaillées incluent les trois métriques.
"""),
        code('''
cross_domain = experiment.summarize_cross_domain()
columns = ["model", "selected_on_btp_cv", "mean_balanced_accuracy", "std_balanced_accuracy"]
columns += [f"balanced_accuracy_{corpus}" for corpus in TEST_CORPORA]
columns += ["ba_ood_avg", "ba_ood_worst"]
display(cross_domain[columns].style.format({col: "{:.3f}" for col in columns[2:]}))
display(cross_domain)
''', "cross_domain"),
        md("""## Lecture des résultats et artefacts
Examiner d'abord la CV BTP, puis le transfert par corpus et les erreurs par classe.
Comparer au notebook 07 avec les mêmes données, filtres, folds et seed. Une différence
de résultat entre TF-IDF et Qwen ne démontre pas à elle seule l'absence de raccourcis lexicaux.

Sous `output/tfidf_baseline/` :
- `config_resolved.json`, `data_summary.csv`, `tfidf_corpus_statistics.csv`, `selection.json` : paramètres et traçabilité ;
- `cv/` : scores, affectations des folds et prédictions hors fold par modèle ;
- `models/` : vectoriseur, vocabulaire/IDF et trois classifieurs finaux ;
- `targets/<corpus>/models/<modèle>/` : prédictions, probabilités, métriques, rapports et matrices ;
- `figures/`, `targets/<corpus>/figures/` : figures PNG ;
- `cross_domain_generalization.csv` : synthèse de toutes les métriques.

Le cache vérifie les paramètres, les fichiers de données, les stopwords et les versions
du calcul. Un changement cible invalide son évaluation ; un changement TF-IDF invalide
la CV et les modèles finaux. Les figures et synthèses sont régénérées.

Cette expérience ne réalise pas de recherche automatique d'hyperparamètres ni de
multi-seeds. Le bootstrap cible est une estimation d'incertitude conditionnelle à
ce run unique. Aucun résultat n'est présupposé avant l'exécution.
"""),
        code('''
artifacts = [path.relative_to(OUTPUT_DIR).as_posix() for path in OUTPUT_DIR.rglob("*") if path.is_file()]
print(f"{len(artifacts)} fichiers disponibles dans {OUTPUT_DIR}")
display(pd.DataFrame({"artefact": sorted(artifacts)}).head(30))
''', "artifacts"),
    ])
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "file_extension": ".py"}
    nbf.validate(nb)
    return nb


def main():
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), NB_PATH)
    print(NB_PATH)


if __name__ == "__main__":
    main()
