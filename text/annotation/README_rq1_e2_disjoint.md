# RQ1 — M3 + E2-B, entraînement disjoint sans cross-fitting

Depuis `SAFER/` sur le cluster :

```bash
sbatch text/jobs/rq1_e2_disjoint_launch.sh
```

Le lanceur soumet un précontrôle, un essai GPU, 18 tâches GPU (`6 cibles × 3 graines`, tableau `0-17%2`) puis une synthèse. Il utilise le paquet GOLD RQ1 privé trouvé par `_rq1_final_env.sh`, ainsi que l'environnement LoRA `.venv1` maintenu par le runtime existant. Le paquet doit contenir `manifest.json`, `complete.json` et les six dossiers `folds/target_<corpus>/`. Le précontrôle vérifie l'empreinte, les partitions et au moins 40 GiB libres pour les deux tâches simultanées. Ce seuil peut être configuré par `RQ1_E2D_MIN_FREE_GIB` après vérification de l'espace requis sur le cluster.

Pour chaque cible et graine, les cinq corpus sources conservent les partitions RQ1 `train`/`val`; le `test` cible est strictement inchangé. Les récits `train` sont regroupés par texte normalisé puis répartis de façon reproductible entre 80 % pour SaT+LoRA et 20 % pour E2. Aucun récit de la cible ou de la validation source n'entre dans ces entraînements. Le même M3 partagé fournit les prédictions `M3_shared` et les caractéristiques de `M3_plus_E2_B`. E2-B reçoit uniquement les états contextuels des tokens voisins de la coupure candidate (256 dimensions), sans probabilité ou position en entrée. La probabilité SaT demeure nécessaire à la génération des candidats.

Un second `M3_full`, entraîné sur tous les récits `train` sources, sert de comparateur pratique. Les checkpoints M3 sont choisis sur une validation **interne à leur seul ensemble autorisé**, puis réentraînés pour l'époque choisie. Le seuil de M3, ainsi que le checkpoint, le seuil candidat et la pénalité E2, sont choisis sur les **mêmes validations sources RQ1**. La cible est évaluée une seule fois après ces choix.

Les sorties privées sont sous `${SCRATCH:-$HOME}/safer_sat_rq1_e2_disjoint/runs/rq1_e2_disjoint_v1/` (modifiable par `RQ1_E2D_ROOT`). Chaque `target_<corpus>/seed_<graine>/` conserve les adaptateurs, pertes, sélections, caractéristiques, prédictions et métriques. `summary/per_target_seed.csv`, `summary/mean_scores.csv`, `summary/paired_gains.csv` et `summary/paired_bootstrap.csv` rassemblent les résultats; la synthèse refuse une rotation manquante. Les IC du dernier fichier sont des bootstraps appariés de 2 000 tirages de récits, calculés séparément par graine; ils ne représentent pas la variabilité entre entraînements. Les caches sont liés au GOLD et au code par `binding.json`, les textes et offsets sont vérifiés à l'extraction et les tâches terminées sont contrôlées avant reprise.

Dans le tableau principal, on comparera `M3_full` et `M3_plus_E2_B` à M0–M2 sur les mêmes 2 690 récits tests. `M3_shared` est le témoin apparié qui isole l'apport du raffineur. Les scores M0–M2 existants ne sont pas recalculés dans cette campagne. Une comparaison M3 pleine taille contre M3+E2-B mesure le *système complet* : elle combine le changement de répartition des données et le raffineur. Les scores du seul pilote BTP antérieur ne doivent pas servir à régler cette campagne.
