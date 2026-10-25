# Cinq rôles sur unités factuelles annotées

Depuis la racine `SAFER/` :

```bash
sbatch text/jobs/role_qwen3_lodo.sh
```

Le lanceur soumet successivement le précontrôle des données (CPU), un essai des trois méthodes adaptées (GPU), les deux baselines sur les six cibles (CPU), un array de 54 entraînements (GPU, `%3`), puis la synthèse (CPU). Le premier `sbatch` ne réalise aucun entraînement lui-même. Les tâches suivantes ne démarrent que si l'étape précédente réussit. Une tâche complète est ignorée lors d'une relance après vérification de ses fichiers.

Entrées requises sur le cluster : `text/.venv`, les six CSV de `text/dataset/narrative_only_v1/`, les six CSV originaux `text/dataset/data_*.csv`, `annotation/lodo_sat_transfer/narrative_only_lodo_v1/`, `text/embeddings/gold_v3/qwen3/` et `text/macro_transfer/stop_metier.txt`. Le précontrôle compare les empreintes des CSV et des plis, les 147 487 unités, les textes/rôles/identifiants de l'export d'embeddings et les cinq classes de chaque validation source. L'essai GPU compare aussi les vecteurs exportés au Qwen3 épinglé.

Les sorties vont dans `$SCRATCH/safer_role_qwen3_narrative_lodo_3epoch_v1/` (ou `ROLE_QWEN3_RUN_ROOT`). `results/scores_by_target.csv` contient les moyennes et écarts types sur trois graines pour CE, SupCon et SoftTriple; les baselines n'ont qu'une évaluation par cible. `results/macro_by_seed.csv` puis `results/macro_six_targets.csv` calculent les macro-moyennes par graine et leur dispersion. Les prédictions, matrices de confusion, courbes de sélection et empreintes sont gardées dans les dossiers de tâche. Seul le meilleur checkpoint de chaque tâche est conservé pendant ses trois époques; il est supprimé après rechargement, prédiction et vérification des sorties.

La sélection utilise la moyenne non pondérée des balanced accuracy sur les cinq corpus sources. Le corpus cible entier intervient seulement après cette sélection. En cas d'égalité, le plus petit `C` puis l'époque la plus précoce sont retenus. CE ne possède pas de `C` et sélectionne uniquement son époque.
