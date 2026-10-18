# RQ1 SaT-LoRA : choisir le checkpoint sur le F1 exact des sources

Cette relance réutilise **sans modification** le bundle privé
`sat12l_sm_gold_v3_rq1` et les mêmes partitions RQ1. Pour chaque domaine cible,
les cinq autres corpus fournissent l'entraînement (80 %) et la validation
source (20 %). Aucun récit cible n'entre dans le choix du checkpoint ou du
seuil. L'évaluation du test cible ne commence qu'après l'écriture de
`checkpoint_selection.json`.

L'entraînement utilise toujours le trainer officiel de la copie versionnée
`wtpsplit`, `sat-12l-sm`, une graine et la configuration LoRA existante. Seule la
conservation des checkpoints change : `save_steps=500` et
`save_total_limit=null`. L'adaptateur final est également candidat, car la fin
des 30 époques ne tombe pas nécessairement sur un pas multiple de 500.
La perte et les métriques internes du trainer restent journalisées ; elles ne
servent pas à choisir le checkpoint de l'étude.

Pour chaque checkpoint, le script extrait ses poids LoRA et sa tête du
`pytorch_model.bin` officiel, recharge cet adaptateur dans SaT, calcule les
probabilités des **mêmes récits de validation source**, puis évalue toute la
grille de seuils RQ1. Le critère est la moyenne non pondérée des cinq F1 de
segments exacts, avec espaces périphériques neutralisés. En cas d'égalité,
le seuil le plus proche de 0,25 est préféré, puis le checkpoint le plus
précoce, puis le plus petit seuil. M1 choisit son seuil sur cette même
validation. Une fois le couple checkpoint–seuil M2 fixé, le script produit
les prédictions et les scores M0/M1/M2 sur le corpus cible.

Le trainer officiel sauvegarde un état complet dans ses checkpoints, mais son
chargeur de reprise cherche des dossiers d'adaptateurs. Le petit lanceur
`run_sat_lora_resume.py` ne modifie que cette lecture : il charge strictement
`pytorch_model.bin`, puis le trainer reprend aussi l'optimiseur, le scheduler,
les états aléatoires et son compteur de pas. Le smoke test passe de l'étape 1
à l'étape 2 avant d'autoriser la campagne.

Depuis la racine `SAFER/` sur le cluster, après `git pull`, avec le bundle v3
déjà transféré et le runtime `.venv1` existant :

```bash
sbatch text/jobs/sat_lodo_gold_v3_checkpoint_launch.sh
```

Le lanceur soumet l'essai GPU, les six rotations après sa réussite et la
synthèse après la réussite du tableau. Les identifiants des jobs sont affichés
dans `slurm-sat-v3-ckpt-launch-<jobid>.out`. Il faut que le cluster autorise
`sbatch` depuis un job Slurm.

Pour soumettre les étapes séparément, commencer par :

```bash
sbatch text/jobs/sat_lodo_gold_v3_checkpoint_smoke.sh
```

Quand le journal confirme `Checkpoint export and optimizer-state resume passed` :

```bash
sbatch text/jobs/sat_lodo_gold_v3_checkpoint_gpu.sh
```

Chaque tâche GPU efface maintenant son propre dossier `target_<corpus>` avant
de commencer. Cela supprime aussi ses anciens checkpoints, prédictions et
scores, y compris si l'exécution précédente était terminée ; les cinq autres
corpus et le smoke test restent intacts. Pour relancer uniquement la tâche 1
(métallurgie) depuis `SAFER/` :

```bash
sbatch --array=1 text/jobs/sat_lodo_gold_v3_checkpoint_gpu.sh
```

Le tableau complet `0-5%2` efface et recalcule les six dossiers cibles.

Les six rotations forment un tableau `0-5%2`, avec deux GPU simultanés au
maximum. Quand les six fichiers `target_*/metrics.json` existent :

```bash
sbatch text/jobs/sat_lodo_gold_v3_checkpoint_summary.sh
```

Les sorties restent **séparées** des exécutions précédentes dans
`${SCRATCH:-$HOME}/safer_sat_lodo/runs/sat12l_sm_gold_v3_rq1_checkpoint_selection_v1/`.
Pour chaque cible : `checkpoint_validation.csv` contient le F1 source de chaque
couple checkpoint–seuil, `checkpoint_selection.json` fixe le choix avant le
test, `source_validation_M1.csv` donne M1, `metrics.json` et `predictions/`
contiennent le test cible, et `loss_history.csv` conserve les pertes. Le
fichier global `checkpoint_selection_summary.csv` résume les choix et
`summary.csv` les scores cibles. Les grands checkpoints complets sont gardés
pour audit : prévoir plusieurs gigaoctets par rotation et un espace de travail
ayant plusieurs dizaines de gigaoctets libres. Une relance avec le job GPU
repart de zéro pour chaque tâche soumise ; le script Python invoqué directement
sans cette remise à zéro conserve son contrôle de provenance et sa reprise.
