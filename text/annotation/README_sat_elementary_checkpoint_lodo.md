# Relance SaT–LoRA sur les unités élémentaires avec choix de checkpoint

Cette relance conserve le GOLD `narrative_v3`, les six rotations LODO et les
exemples « faits + `excluded_spans` » préparés pour
`sat_elementary_lodo.py`. Elle change la sélection du modèle : comme l'ancien
M3, elle exporte les checkpoints tous les 1 000 pas et choisit **checkpoint
et seuil conjointement** sur les cinq corpus sources. Le critère est ici le
F1 exact des unités élémentaires, car c'est la tâche de segmentation actuelle.
Les récits du corpus cible ne servent qu'à l'évaluation finale. Une seule
graine est utilisée ; il n'y a pas de réentraînement après la sélection.

Le premier run élémentaire utilisait les poids finaux. Cette relance écrit
dans un dossier distinct et ne le remplace pas. Elle réutilise les deux
paquets déjà placés sur le cluster :

* `sat12l_sm_gold_v3_rq1/` : plis, récits GOLD et SaT initial épinglé ;
* `sat12l_sm_elementary_v1/` : exemples d'entraînement faits + exclus.

Il n'est pas nécessaire de régénérer les `training.pth`. Après `git pull`,
depuis la racine `SAFER/` sur le cluster :

```bash
sbatch text/jobs/sat_elementary_checkpoint_launch.sh
```

Le lanceur soumet un smoke GPU, un tableau de six tâches GPU (`0-5%3`, au
plus trois simultanées), puis un résumé CPU, tous liés par `afterok`. Il y a
une graine par rotation, et non plusieurs graines. Le smoke contrôle
l'export d'un checkpoint en adaptateur LoRA et la reprise de l'optimiseur.
Les fichiers vont dans
`${SCRATCH:-$HOME}/safer_sat_lodo/runs/sat12l_sm_elementary_checkpoint_v2/`.
Un chemin personnalisé pour le paquet élémentaire peut être fourni avec
`SAT_ELEMENTARY_PREPARED`; le paquet RQ1 utilise `SAT_LODO_BUNDLE`.

Pour chaque domaine cible, `checkpoint_validation.csv` contient les scores
source de chaque checkpoint et seuil ; `selection.json` désigne le couple
retenu ; `metrics.json` et `predictions.jsonl` concernent uniquement le test
cible. `summary.csv` rassemble les six domaines. Les états complets de
l'optimiseur sont supprimés uniquement après vérification des poids choisis,
des prédictions et des métriques ; les adaptateurs LoRA compacts et courbes
de validation restent disponibles. Le nouveau F1 ne peut pas être attribué
au seul ajout des exclusions, puisque l'ancien protocole employait aussi
une autre partition et un autre critère de sélection.
