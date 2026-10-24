# Segmentation SaT : mono-source contre multisource à volume égal

Cette campagne compare, sur les 30 transferts source-cible, un modèle SaT+LoRA
mono-source déjà entraîné à un nouveau modèle entraîné sur les cinq corpus hors
cible. Pour chaque paire, le nombre de caractères transmis au trainer est égal
à 1 % près, le nombre de mises à jour de l'optimiseur est identique, et la
validation est celle de l'unique corpus source. Le test cible reste figé.

La comparaison fine est faite **en local**, où les six résultats mono-source
existent déjà. Aucun ancien adaptateur mono-source ni dossier `source_<corpus>`
ne doit être copié sur le cluster. Les deux méthodes n'ont pas nécessairement
le même nombre de récits, d'unités ou de sous-tokens ; les effectifs sont
consignés. Une seule graine LoRA est utilisée ; l'IC bootstrap apparié ne
mesure pas la variabilité entre entraînements.

## Préparer la référence sur le PC

Depuis la racine de `SAFER` sous PowerShell, avec les résultats mono-source
locaux encore présents :

```powershell
.\text\.venv\Scripts\python.exe text\scripts\rq1_seg_equal_volume.py export-reference `
  --source-bundle safer_sat_indomain\bundle\sat12l_sm_gold_v3_indomain_v1 `
  --single-runs safer_sat_indomain\runs\sat12l_sm_gold_v3_indomain_v1 `
  --reference safer_sat_indomain\bundle\sat12l_sm_gold_v3_indomain_v1\equal_volume_reference.json
```

La commande audite les six anciens entraînements, leurs adaptateurs retenus,
leurs empreintes et leurs métriques par récit. Elle écrit seulement un fichier
de référence portable d'environ 21 Ko dans le paquet privé GOLD. Ce fichier
est ignoré par Git avec le reste du paquet et doit être transféré avec lui.

## Lancer sur le cluster

Après synchronisation du **code** et du seul dossier privé
`safer_sat_indomain/bundle/sat12l_sm_gold_v3_indomain_v1/` (avec
`equal_volume_reference.json`), depuis `SAFER/` :

```bash
sbatch text/jobs/rq1_seg_equal_volume_launch.sh
```

Le lanceur recherche ce paquet sous `$SCRATCH`, `$HOME` ou `SAFER/`. Il utilise
le clone versionné de `wtpsplit` et recrée si nécessaire `.venv1` dans
`$SCRATCH/safer_sat_lodo/runtime_gold_v3/`. Il enchaîne précontrôle, essai GPU,
30 tâches `0-29%2`, puis un audit final. L'adaptateur de l'essai GPU sert de
gabarit technique pour exporter les checkpoints des nouveaux entraînements ;
ses poids ne sont pas chargés pour initialiser ces entraînements.

Les sorties sont dans `$SCRATCH/safer_sat_equal_volume_v2/` (ou `$HOME` si
`$SCRATCH` est absent). `multi_only_complete.json` signale que les 30 sorties
multisources ont été vérifiées. Le cluster ne calcule aucun gain sur les
anciens résultats mono-source, qu'il ne possède pas.

## Comparer après rapatriement

Rapatrier le dossier `safer_sat_equal_volume_v2/` dans `SAFER/` sur le PC.
Depuis la racine du dépôt :

```powershell
.\text\.venv\Scripts\python.exe text\scripts\rq1_seg_equal_volume.py compare-local `
  --source-bundle safer_sat_indomain\bundle\sat12l_sm_gold_v3_indomain_v1 `
  --reference safer_sat_indomain\bundle\sat12l_sm_gold_v3_indomain_v1\equal_volume_reference.json `
  --single-runs safer_sat_indomain\runs\sat12l_sm_gold_v3_indomain_v1 `
  --run-root safer_sat_equal_volume_v2
```

Cette commande refuse un modèle manquant, un changement d'empreinte, des
identifiants de test différents ou un décalage des comptes GOLD. Elle produit
`pair_results.csv`, `target_summary.csv`, `transfer_matrix_delta.csv` et
`.svg`, puis `RESULTS.json` avec 2 000 bootstraps appariés par récit. Les
résultats historiques restent intacts.
