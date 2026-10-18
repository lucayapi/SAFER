# Campagne RQ1 : segmentation factuelle sur tests communs

Cette campagne ne traite que la segmentation. Le GOLD `narrative_v3` contient
17 927 récits et 147 487 unités ; les six partitions 70/15/15 sont figées par
le bundle mono-source déjà contrôlé. Les 2 690 récits de test sont communs à
PySBD, SaT initial, SaT calibré et SaT + LoRA. Les anciens scores sont inventoriés
dans [legacy_rq1.md](legacy_rq1.md), mais ne sont pas mélangés aux nouveaux.

## Avant l'envoi du code

Depuis `SAFER/` sous Windows, avec `text/.venv` :

```powershell
.\text\.venv\Scripts\python.exe text\scripts\rq1_final.py audit-single --output annotation\rq1_final_v1\single_source_audit.json
.\text\.venv\Scripts\python.exe text\scripts\rq1_final.py prepare
.\text\.venv\Scripts\python.exe text\scripts\rq1_final.py check
.\text\.venv\Scripts\python.exe text\scripts\rq1_final_report.py --baseline-only
```

L'audit doit lister les six sources dans `reusable` et zéro dans `failed`.
Le bundle local privé est `annotation/rq1_final_v1/bundle/` ; il n'est pas
commité. Le GOLD, les tests et les anciens résultats restent à leur place.
La dernière commande calcule localement PySBD, SaT initial et les six
calibrations M2 sur validation source, puis écrit les métriques et prédictions
dans `annotation/rq1_final_v1/local_baselines/`.

Le cluster dispose normalement déjà du paquet privé
`${SCRATCH:-$HOME}/safer_sat_indomain/bundle/sat12l_sm_gold_v3_indomain_v1/`
qui a servi aux modèles mono-source. Le job CPU le contrôle et reconstruit le
nouveau paquet LODO à partir de ce paquet : aucun transfert supplémentaire
n'est nécessaire s'il est encore présent. Sinon, il faut restaurer ce paquet
privé **hors Git** avant de soumettre.

## Cluster Slurm

Après `git pull`, depuis la racine `SAFER/` :

```bash
sbatch text/jobs/rq1_final_launch.sh
```

Ce seul lancement soumet un précontrôle CPU, un essai GPU, puis six rotations
LODO `0-5%2` et un résumé CPU dépendant de la réussite de toutes les tâches.
Un GPU est demandé par rotation ; au plus deux tâches GPU tournent ensemble.
La préparation refuse moins de 12 GiB libres sur le système de fichiers et
affiche aussi le quota lorsqu'il est disponible. Le quota personnel peut être
plus restrictif que l'espace indiqué par `df`.

Les journaux Slurm sont `slurm-rq1-final-*.out/.err` dans le dossier de
soumission. Les sorties privées sont dans :

```text
${SCRATCH:-$HOME}/safer_sat_rq1_final/bundle/rq1_segmentation_final_v1/
${SCRATCH:-$HOME}/safer_sat_rq1_final/runs/rq1_segmentation_final_v1/
```

Chaque rotation sélectionne checkpoint et seuil sur la validation des cinq
sources, en maximisant leur moyenne macro des F1 exacts. La cible n'entre dans
aucune sélection. La grille de seuils, le modèle et le trainer officiel sont
figés par le code ; les checkpoints sont espacés de 1 000 pas. Pendant
l'entraînement, chaque checkpoint complet est exporté en petit adaptateur
LoRA, puis les anciens états complets sont supprimés ; seul le plus récent
reste disponible pour une reprise. Tous les adaptateurs exportés sont évalués
sur validation source, avec le modèle final comme candidat supplémentaire.
Après production et contrôle des prédictions, seul l'adaptateur sélectionné
est conservé, avec les courbes de validation, pertes et empreintes ; les états
complets sont supprimés. Les six entraîneurs mono-source ne sont pas relancés.
Ce nettoyage progressif s'applique aux tâches soumises avec cette version du
code ; il ne modifie pas un job Slurm déjà en cours. Il ne faut pas supprimer
manuellement le dernier checkpoint complet d'une tâche interrompue avant sa
reprise.

En cas d'échec d'une tâche du tableau, inspecter son `.err`, corriger la cause,
puis soumettre seulement son indice :

```bash
sbatch --array=INDICE text/jobs/rq1_final_gpu.sh
```

Après une reprise ciblée, lancer le contrôle final :

```bash
sbatch text/jobs/rq1_final_summary.sh
```

La reprise vérifie l'identité du paquet, du code et des poids. Elle ne réinitialise
pas les anciens dossiers. `cluster_complete.json` n'est écrit que lorsque les
six rotations et leurs prédictions sont présentes.

## Rapport local

Copier **hors Git** le contenu du dossier compact
`runs/rq1_segmentation_final_v1/report_transfer/` du cluster vers
`SAFER/safer_sat_rq1_final/runs/rq1_segmentation_final_v1/`. Ce dossier
contient les prédictions, les sélections, les empreintes et les adaptateurs
retenus ; les gros checkpoints intermédiaires restent sur le cluster. Puis :

```powershell
.\text\.venv\Scripts\python.exe text\scripts\rq1_final_report.py
```

Le script refuse une rotation manquante ou une divergence de GOLD, de test,
de sélection ou de prédictions. Il écrit dans
`annotation/rq1_final_v1/results/` les prédictions à offsets, les tableaux
CSV/LaTeX, la matrice de transfert, le rappel exact par rôle GOLD, les catégories
d'erreurs et les IC bootstrap appariés des gains M1−M0, M2−M1 et M3−M2.
La diagonale mono-source est `N/A`.

Le F1 exact neutralise les espaces périphériques. Le Soft Span F1 emploie
l'IoU en caractères et un appariement optimal un-à-un de tout chevauchement
positif ; `IoU ≥ 0,75` est une sensibilité distincte. Une seule graine LoRA
est utilisée : les IC bootstrap ne mesurent pas la variabilité entre graines.
Le GOLD a été confirmé par la chercheuse à partir de préannotations IA et
n'est pas une double annotation humaine indépendante. L'entraînement officiel
reconstruit les unités factuelles, alors que l'évaluation utilise les récits bruts.

