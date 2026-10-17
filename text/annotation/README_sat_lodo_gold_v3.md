# RQ1 : comparaison équitable de SaT initial, calibré et adapté

Cette expérience utilise le `gold_snapshot.jsonl` **narrative_v3 confirmé** de
`annotation/rq1_sat_v3_20261017/`, avec 17 927 récits et 147 487 unités. Elle
est distincte des anciennes expériences `sat12l_sm_lodo_v*`, faites sur les
préannotations v2. Les résultats anciens restent des archives et ne sont pas
comparables directement aux nouveaux.

Pour chacune des six rotations Leave-One-Domain-Out, les cinq domaines sources
reprennent **exactement la partition RQ1** : 80 % de chaque source servent à
entraîner le LoRA ; les 20 % `source_holdout` servent à choisir **séparément**
les seuils de M1 et M2, avec les mêmes récits, les mêmes unités et la même
grille. Le domaine cible n'intervient dans aucune de ces décisions et est
évalué en entier après sélection. M0 est le modèle original à 0,25 ; M1 est
le modèle original recalibré sur cette validation commune ; M2 est le modèle
LoRA recalibré sur cette validation commune. Le critère de sélection est la
moyenne non pondérée des cinq F1 micro par source pour les unités exactes,
espaces périphériques neutralisés. Une égalité est départagée par la proximité
avec 0,25, puis par le seuil le plus faible. Les trois modèles sont testés
sur les **mêmes récits cibles**.

Le modèle de base et le tokenizer sont épinglés aux révisions enregistrées par
RQ1. Le script refuse les fuites entre partitions, les données altérées, et
une valeur M0 qui ne reproduit pas le résultat RQ1. Les unités factuelles du
jeu d'entraînement sont des extraits littéraux. Les parties exclues ne sont
pas transformées en faits. Les récits sans unité ou avec saut de ligne *dans*
une unité ne sont pas donnés au trainer officiel, mais restent dans les jeux
de validation et de test ; le manifeste compte ces cas. L'entraînement utilise
la même configuration LoRA pour chaque rotation (30 époques prévues, une
graine, dernier adaptateur sauvegardé, sans sélection de checkpoint sur la
cible). Les journaux de pertes sont exportés séparément.

## Préparer et transférer les données privées

Depuis la racine de SAFER sur le poste local :

```powershell
.\text\.venv\Scripts\python.exe text\scripts\sat_lodo_gold_v3.py prepare
.\text\.venv\Scripts\python.exe text\scripts\sat_lodo_gold_v3.py check-bundle
```

Le dossier obtenu est
`annotation/lodo_sat_transfer/sat12l_sm_gold_v3_rq1/`. Il est ignoré par Git :
copier **ce dossier entier**, manifeste et six sous-dossiers `folds` compris,
vers `$SCRATCH/safer_private/sat12l_sm_gold_v3_rq1/` sur le cluster (ou
`$HOME/safer_private/sat12l_sm_gold_v3_rq1/` si `SCRATCH` n'est pas défini). Ne pas
copier seulement `training.pth`, car l'évaluation utilise aussi les textes
GOLD et les probabilités SaT initiales épinglées. Le code de `text/` et la
copie versionnée de `text/vendor/wtpsplit/` doivent être présents sur le
cluster via le dépôt. Aucun récit privé ne doit être commité.
Le job accepte aussi le dossier déjà copié dans
`SAFER/annotation/lodo_sat_transfer/sat12l_sm_gold_v3_rq1/` ; il affiche le
chemin effectivement choisi. Si les données sont ailleurs, définir
`SAT_LODO_BUNDLE` vers le dossier contenant `manifest.json` et `complete.json`.

## Lancer uniquement avec Slurm

Depuis la racine `SAFER/` sur le cluster, après transfert des données, on peut
soumettre les trois étapes d'un coup avec dépendances :

```bash
smoke=$(sbatch --parsable text/jobs/sat_lodo_gold_v3_smoke.sh)
train=$(sbatch --parsable --dependency=afterok:${smoke} text/jobs/sat_lodo_gold_v3_gpu.sh)
sbatch --dependency=afterok:${train} text/jobs/sat_lodo_gold_v3_summary.sh
```

Le smoke installe l'environnement LoRA de cette expérience dans
`$SCRATCH/safer_sat_lodo/runtime_gold_v3/.venv1/` (ou sous `$HOME`) et ne
réutilise pas le `venv` précédent du cluster. Les commandes PowerShell plus
haut utilisent seulement le `.venv` **local Windows** pour préparer les données.
Le smoke entraîne ensuite un pas et vérifie que l'adaptateur peut être
rechargé. Le tableau GPU `0-5%2` lance six
rotations avec au plus deux tâches simultanées. Le résumé CPU démarre
uniquement quand toutes les rotations ont réussi. Les chemins privés sont
fixés dans `text/jobs/_sat_lodo_gold_v3_env.sh` ; aucune variable ne doit être
saisie pour le cas standard. Les journaux Slurm `slurm-sat-gold-v3*.out/.err`
sont écrits dans le dossier depuis lequel `sbatch` a été appelé.

Les fichiers résultants sont dans
`$SCRATCH/safer_sat_lodo/runs/sat12l_sm_gold_v3_rq1/` (ou sous `$HOME` selon
la même règle) : pour chaque cible,
`config.json`, `training_done.json`, `loss_history.csv`, l'adaptateur sous
`trainer/safer_lodo/fr/`, `source_validation_thresholds.csv`,
`document_metrics.csv`, `metrics.json` et les trois jeux de segments dans
`predictions/`. À la racine du dossier, `summary.csv`, `gain_ci95.csv` et
`summary.json` réunissent les résultats. Le bootstrap est apparié par récit ;
les scores principaux incluent tous les récits cibles, et un diagnostic
distinct est fourni pour les références sans signalement.

Les préannotations v3 ont été confirmées par l'utilisatrice ; cette procédure
ne constitue pas une mesure indépendante d'accord entre annotateurs. Toute
présentation scientifique doit indiquer cette provenance et les cas signalés.
