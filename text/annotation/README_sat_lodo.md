# SaT-LoRA : six rotations LODO exploratoires

Ce protocole entraîne `sat-12l-sm` sur cinq corpus sources et évalue le sixième,
répété pour les six cibles. Les seuils de M1 (SaT initial) et M3 (LoRA) sont
choisis uniquement sur la validation des sources. M0 utilise le seuil par défaut.
Les références restent des **préannotations IA**, et non un gold humain validé.

## Données utilisées

Le bundle privé `sat12l_sm_lodo_v3` reprend toutes les préannotations directes
`narrative_v2` qui possèdent au moins une unité et des extraits exacts, ordonnés
et sans chevauchement. Les champs `issues` et `uncertain` ne sont plus des motifs
d'exclusion. Les annotations produites par les anciennes règles locales restent
exclues, conformément au choix antérieur de n'utiliser que l'annotation directe.

Le « contrôle qualité » est la file
`annotation/audit/validation_queue_20261016/` : tout récit cité dans
`cases_to_validate.jsonl` ou `source_quality_warnings.jsonl` est écarté de
l'entraînement **et de l'évaluation**. Le manifeste conserve les empreintes
SHA-256 de ces deux listes et les effectifs par corpus. Un signal de la file
reste un cas à examiner, pas nécessairement une erreur confirmée.

Lorsqu'un récit contient du texte informatif hors des unités annotées, seule la
suite de ses unités sert à l'entraînement : aucun faux segment n'est inventé
pour le passage manquant. Cette transformation réduit le contexte du modèle
pour ces récits. Les retours à la ligne sont remplacés par des espaces dans
la seule copie d'entraînement. Les récits et offsets d'origine restent intacts
dans `train.jsonl`, `val.jsonl` et `test.jsonl` pour l'évaluation. Les récits
sans unité factuelle ne peuvent pas former un exemple du trainer officiel et
sont comptés séparément. Les données des six rotations et le partage 80/20
reproductible sont décrits dans `manifest.json`.

Préparation effectuée sur le PC, depuis `SAFER/` :

```powershell
.\text\.venv\Scripts\python.exe text\scripts\sat_lodo.py preflight
.\text\.venv\Scripts\python.exe text\scripts\sat_lodo.py prepare --out annotation\lodo_sat_transfer\sat12l_sm_lodo_v3
.\text\.venv\Scripts\python.exe text\scripts\sat_lodo.py check-bundle --bundle annotation\lodo_sat_transfer\sat12l_sm_lodo_v3
```

Le dossier `annotation/` est ignoré par Git. **Ne commitez pas les récits.**
Transférez séparément `annotation/lodo_sat_transfer/sat12l_sm_lodo_v3/` vers
`$SCRATCH/safer_private/sat12l_sm_lodo_v3/` sur le cluster, ou vers
`SAFER/annotation/lodo_sat_transfer/sat12l_sm_lodo_v3/`. Les anciens bundles
`v1`/`v2` sont des archives : ils ne sont pas convertibles en `v3`, car les
partitions et les données ont changé.

## Environnement du cluster

Le code `text/vendor/wtpsplit/` est une copie versionnée du trainer officiel,
révision `3ef928c8e163d11dc6f6336ad610bff2ffe0bda4`. Le job vérifie
ses empreintes, le GPU, les versions Python, la configuration et le bundle.
Le runtime isolé installé lors de l'essai GPU précédent est réutilisé depuis
`${SCRATCH:-$HOME}/safer_sat_lodo/runtime/`. Il comprend PyTorch 2.5.1 CUDA 12.4,
les dépendances de `text/annotation/requirements_sat_lodo.txt`, ainsi que
`wtpsplit` et `adapters`. Aucun clone ni installation manuelle n'est nécessaire
si ce runtime est déjà prêt. Aucune journalisation W&B n'est activée.

La configuration commune est dans `text/annotation/sat_lodo_lora_template.json` :
`sat-12l-sm`, 30 époques, taux `3e-4`, LoRA `r=16` et `alpha=32`, lot GPU 8
avec accumulation 8, `subsample=null`. `use_auxiliary=false` garde la tête
préentraînée à une sortie. Les sorties sont privées, dans
`${SCRATCH:-$HOME}/safer_sat_lodo/runs/sat12l_sm_lodo_v3/`.

## Lancer directement les six adaptations

Après `git pull` du code et transfert du **bundle v3** sur le cluster, depuis
la racine `SAFER/` :

```bash
sbatch text/jobs/sat_lodo_gpu.sh
```

Le tableau Slurm `0-5%2` lance six tâches avec au plus deux GPU en même temps.
Il n'exige plus de nouveau job `smoke` pour ce bundle ; il utilise le runtime
GPU déjà installé et effectue ses précontrôles avant chaque entraînement.
Depuis `SAFER/text/`, la commande est `sbatch jobs/sat_lodo_gpu.sh`.
Les journaux `slurm-sat-lodo-<job>_<index>.out` et `.err` apparaissent dans le
dossier de soumission. Une relance reprend les checkpoints existants de `v3`.

Chaque cible produit son adaptateur, `loss_history.csv`, `metrics.json`, les
courbes de seuils et un cache SQLite de prédictions. Les pertes sont aussi dans
`trainer/trainer_state.json` et visibles pendant l'entraînement dans `.out`.
Après les six tâches :

```bash
sbatch text/jobs/sat_lodo_summary.sh
```

`summary.csv` compare M0, M1 et M3 par corpus, puis leur moyenne macro.
Ces scores sont exploratoires et ne doivent pas être présentés comme des
performances sur un gold humain indépendant.

## Vérifier la borne basse des seuils M1

Le rapport d'article normalise les espaces périphériques. M1 atteint la borne
`0,01` dans cinq rotations sur six : cela appelle une vérification sur les
**validations sources uniquement**. Le script
`text/scripts/sat_lodo_low_thresholds.py` ajoute `0,001`, `0,0025`,
`0,005` et `0,0075`, contrôle `0,01` contre le cache initial, puis choisit
le meilleur seuil de SaT initial par moyenne macro des F1 de segments
neutralisés des cinq sources. Aucun fichier `test.jsonl` cible n'est ouvert.
Les prédictions et le rapport sont séparés sous
`safer_sat_lodo/resultats_articles/validation_seuils_inferieurs/` ;
le tableau d'article existant et les prédictions cibles ne sont pas modifiés.
Une reprise saute les récits déjà calculés après vérification des empreintes.

Sur le cluster, avec le bundle v3 et les six caches de prédictions déjà en
place, depuis `SAFER/` :

```bash
sbatch text/jobs/sat_lodo_low_thresholds.sh
```

Le journal Slurm montre la barre `tqdm` et les seuils retenus. Pour un essai
local limité à un récit, depuis `SAFER/` :

```powershell
.\text\.venv\Scripts\python.exe text\scripts\sat_lodo_low_thresholds.py run --max-documents 1
```

Cet essai n'écrit pas de courbes finales. Après la vérification source, il
faudra produire de nouvelles prédictions cibles **uniquement si** un seuil
inférieur à `0,01` est effectivement retenu, avant de recalculer les métriques
finales.

Un calcul local complet a été effectué sur les 3 252 récits distincts des
validations sources. BTP, Bois–textile et Service temporaire restent à `0,01`,
Nicollin reste à `0,05`, tandis que Métallurgie et Chimie–plastiques
sélectionnent `0,001` avec des gains source de `0,000163` et `0,001609` en F1.
`0,001` est encore une borne de grille ; aucune amélioration sur les cibles
n'est déduite de ce contrôle. Les détails sont dans
`safer_sat_lodo/resultats_articles/validation_seuils_inferieurs/README.md`.
