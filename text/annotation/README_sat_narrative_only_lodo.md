# SaT + LoRA sur le récit nettoyé

Cette expérience remplace l'ancien entraînement « unité élémentaire » qui
fournissait les faits **et** les spans exclus au trainer. Ici, seuls les faits
annotés servent de segments d'entraînement. Les spans explicitement exclus
sont retirés du récit avant la préparation des plis ; les partitions RQ1
(80 % entraînement / 20 % validation dans les cinq sources et corpus cible
entier en test) sont conservées. Une graine et la configuration SaT + LoRA
historique sont utilisées.

La métrique principale est le **F1 exact normalisé aux bords** : les espaces
et caractères Unicode de ponctuation au début ou à la fin d'un span sont
ignorés. Les caractères internes, y compris la ponctuation interne, doivent
coïncider exactement. Le seuil est choisi par macro-F1 micro par source sur
les cinq corpus de validation ; le corpus cible n'intervient pas dans ce choix.

## Préparation locale

Depuis la racine SAFER :

```powershell
.\text\.venv\Scripts\python.exe text\scripts\sat_narrative_only_lodo.py prepare
.\text\.venv\Scripts\python.exe text\scripts\sat_narrative_only_lodo.py check
.\text\.venv\Scripts\python.exe -m pytest text\tests\test_sat_narrative_only_metrics.py -q -p no:cacheprovider
```

Le paquet à transférer est
`annotation/lodo_sat_transfer/sat12l_sm_narrative_only_v1/`. Il contient les
six plis nettoyés, les données SaT `training.pth`, les hashes et les rapports.
Il n'est pas nécessaire de transférer séparément
`text/dataset/narrative_only_v1/`.

## À avoir sur le cluster

1. Le code SAFER à jour, notamment `text/scripts/sat_narrative_only_lodo.py`
   et les quatre fichiers `text/jobs/sat_narrative_only_*.sh` ainsi que
   `_sat_narrative_only_env.sh`.
2. Le nouveau paquet ci-dessus, copié vers
   `$SCRATCH/safer_private/sat12l_sm_narrative_only_v1/`.
3. Le paquet de base déjà utilisé auparavant :
   `$SCRATCH/safer_private/sat12l_sm_gold_v3_rq1/`, avec les révisions épinglées
   du modèle et le découpage RQ1. S'il est déjà sur le cluster, ne pas le
   recopier.

Depuis `SAFER/` sur le cluster, lancer :

```bash
bash text/jobs/sat_narrative_only_launch.sh
```

Cela soumet un smoke GPU, six tâches LODO (au plus trois simultanées), puis un
résumé CPU dépendant de leur réussite. Les résultats sont écrits sous
`$SCRATCH/safer_sat_lodo/runs/sat12l_sm_narrative_only_v1/`.
