# Raffinement des frontières des faits et des passages exclus

Cette campagne entraîne **un seul raffineur**, avec une graine fixée (13), dans
les six rotations où un corpus est laissé de côté. Le modèle SaT final est ajusté
une fois par rotation, avec sa propre graine fixée et enregistrée. Cinq modèles
SaT hors pli par rotation fournissent les entrées d'apprentissage du raffineur.
Le raffineur
reçoit les représentations du texte à gauche et à droite de chaque coupure
candidate. Ses exemples d'entraînement proviennent de prédictions SaT hors pli
sur les cinq corpus sources. Les bords des unités factuelles et des passages
explicitement exclus sont positifs ; les positions strictement à l'intérieur
de ces intervalles sont négatives ; les zones non annotées restent inconnues.
L'époque et les seuils sont choisis sur la validation source par la moyenne
harmonique du F1 exact des faits et du rappel exact des passages exclus,
calculés d'abord par corpus puis moyennés entre les cinq sources. Aucune
annotation du corpus cible ne sert à l'entraînement ou au choix.

Cette campagne mesure la **délimitation** des passages exclus. Elle n'entraîne
pas encore la décision de les rejeter ni la classification des rôles. Le rappel
des exclusions exactement délimitées ne doit pas être présenté comme un taux
de rejet.

## Lancement sur le cluster

Depuis la racine `SAFER` :

```bash
sbatch text/jobs/rq1_exclusion_boundaries_launch.sh
```

Le lancement enchaîne préparation, deux contrôles GPU, 30 prédictions hors pli
(cinq par corpus cible), six modèles SaT finaux, six entraînements du même
raffineur (un par corpus cible), puis le résumé. Jusqu'à trois tâches GPU
peuvent être exécutées simultanément dans chaque tableau ; le groupe suivant
attend la fin de tout le groupe précédent. Les tâches déjà
terminées dans la campagne SaT source sont réutilisées si leurs empreintes et
leurs fichiers concordent. Pour utiliser volontairement une campagne source
existante, définir `RQ1_E2X_ROOT` avant `sbatch` ; sinon le lancement utilise
un nouveau dossier réservé à cette expérience.

## Éléments nécessaires

- Le dépôt à jour sur le cluster, notamment `text/scripts/`, `text/jobs/`,
  `text/vendor/wtpsplit/` et `text/annotation/requirements_sat_lodo.txt`.
- Le paquet privé GOLD RQ1 complet (`manifest.json`, `complete.json`,
  `rq1_runtime.json`, `folds/`, etc.) dans
  `$SCRATCH/safer_sat_rq1_final/bundle/rq1_segmentation_final_v1/` ou
  `SAFER/annotation/rq1_final_v1/bundle/`. Il n'est pas nécessaire de copier
  les six anciens dossiers `source_<corpus>` d'une autre expérience.
- Les modules du cluster `gcc/8.1.0` et `python/3.10.10`, Slurm, et une GPU
  A100 ou H100. Le lancement prépare ou répare l'environnement dédié
  `$SCRATCH/safer_sat_lodo/runtime_gold_v3/.venv1` ; il n'utilise pas
  `text/.venv`.
- Un accès aux révisions épinglées de `segment-any-text/sat-12l-sm` et
  `FacebookAI/xlm-roberta-base`, soit dans le cache Hugging Face du cluster,
  soit par téléchargement. Les dépendances Python demandent aussi un accès
  aux roues ou à un miroir du cluster lors de la première installation.
- Un volume `SCRATCH` partagé entre les nœuds et assez d'espace libre : le contrôle initial estime environ
  **18 Gio libres** pour les caractéristiques des six rotations et une marge
  calculée pour deux entraînements simultanés. Avec trois tâches GPU en
  parallèle, prévoir au moins **4 Gio supplémentaires** pour les entraînements,
  en plus de la place nécessaire au cache des modèles et à l'environnement
  Python. Choisir `SCRATCH` sur un volume assez grand.

Les résultats de ce lancement se trouvent par défaut dans
`$SCRATCH/safer_sat_rq1_exclusion_boundaries/runs/seed13_v2/summary/` :
`per_domain.csv` et `macro.csv`. Le tableau par domaine
contient aussi le plafond de rappel dû à la disponibilité des deux bornes de
chaque passage exclu dans la grille candidate. Les prédictions et les mesures
détaillées de chaque corpus restent dans `target_<corpus>/`.
Ce lancement ne calcule aucune différence entre méthodes. La comparaison
avec les autres méthodes pourra être faite ensuite sur les mêmes tests, à partir
de leurs prédictions et des résultats de cette campagne.
