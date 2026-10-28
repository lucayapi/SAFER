# Évaluation de bout en bout réellement calculable

Depuis la racine `SAFER/` sur le cluster :

```bash
ROLE_E2E_AFTEROK=9172885:9172887 bash text/jobs/role_narrative_e2e_baselines_launch.sh
```

`ROLE_E2E_AFTEROK` est facultatif. Ici, il évite de faire tourner les six
évaluations GPU en même temps que les arrays CE/Triplet `9172885` et `9172887`. Le lanceur
soumet un précontrôle CPU, un essai GPU du Qwen3 gelé sur quelques segments
réels, un array de **six cibles limité à trois GPU simultanés**, puis une
synthèse CPU. Il ne réentraîne ni SaT ni Qwen3.
Il réajuste seulement les deux régressions logistiques avec
les valeurs de `C` déjà sélectionnées sur les corpus sources.

Pour chaque cible, le job vérifie que les plis, les textes, les prédictions
M0–M3 et les résultats de classification proviennent des mêmes données.
Il reconstruit TF–IDF + LR et Qwen3 gelé + LR à partir des sources, puis
mesure leur concordance avec les prédictions enregistrées sur les **unités
gold**. Un écart supérieur à 1 % arrête la tâche : les poids LR n'ont pas
été conservés et des différences mineures de solveur sont possibles.
Les métriques gold du classifieur réajusté et le nombre d'écarts sont
enregistrés pour éviter de le confondre avec le modèle initial.
Il applique ensuite chaque classifieur aux
segments prédits par M0, M1, M2 et M3, sans lire leur rôle gold pour
prédire. Les spans sont appariés une-à-une à IoU caractère ≥ 0,5 après
la même normalisation des bords que la comparaison de segmentation.
Le F1 sensible au rôle compte seulement les paires ayant aussi le bon rôle.

Les résultats vont dans
`$SCRATCH/safer_role_qwen3_narrative_lodo_3epoch_v1/e2e_baselines_v1/`
(modifiable avec `ROLE_E2E_OUTPUT_ROOT`). Chaque cible conserve
`scores.csv`, `predictions.csv`, `refit_audit.json` et leurs empreintes.
La synthèse produit `results/scores_by_target.csv`,
`results/macro_six_targets.csv` et `results/refit_audit_by_target.csv`.
Ce sont **huit configurations réelles** : quatre segmenters × deux
classifieurs, sur six cibles. Aucun chiffre fictif n'est utilisé.

Les entrées nécessaires sont le paquet
`annotation/lodo_sat_transfer/narrative_only_lodo_v1/`, les prédictions
`$SCRATCH/safer_sat_lodo/runs/narrative_only_lodo_v3_baselines/` et
`$SCRATCH/safer_sat_lodo/runs/narrative_only_lodo_checkpoint_v1/`,
la campagne de rôles `$SCRATCH/safer_role_qwen3_narrative_lodo_3epoch_v1/`,
les six exports Qwen3 et `text/.venv`. Les chemins sont surchargables
par `SAT_NARRATIVE_PREPARED`, `SAT_NARRATIVE_BASELINE_ROOT`,
`SAT_NARRATIVE_RUN_ROOT` et `ROLE_QWEN3_RUN_ROOT`.

**Limite importante.** Les résultats CE, SupCon, SoftTriple et Triplet
enregistrés portent uniquement sur les unités gold. Leurs meilleurs
checkpoints Qwen3 complets sont supprimés après vérification du test.
Leurs CSV de prédictions ne permettent donc pas de prédire les rôles des
segments SaT : un appariement aux unités gold introduirait une information
de référence dans le système et ne serait pas un vrai résultat de bout
en bout. Le précontrôle écrit
`adapted_checkpoint_availability.json` pour documenter la disponibilité
des poids. Pour obtenir le vrai F1 de bout en bout de ces quatre méthodes,
il faut conserver le modèle sélectionné ou calculer ses prédictions sur
les spans M0–M3 avant la suppression du checkpoint, ce qui demanderait
une nouvelle exécution des entraînements déjà terminés.
