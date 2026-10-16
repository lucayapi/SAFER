# SaT-LoRA : six rotations LODO exploratoires

Ce protocole compare `sat-12l-sm` initial (M0), son seuil calibré sur les cinq
corpus sources (M1) et un adaptateur LoRA entraîné sur ces sources puis calibré
sur leur validation (M3). Chaque cible est absente de l'entraînement, de la
validation et du choix du seuil. Une rotation correspond à **une tâche GPU** ;
le tableau Slurm lance les six rotations avec deux tâches simultanées au maximum.

**Nature de la référence :** les unités employées ici sont des préannotations IA
directes, pas des annotations humaines validées. Les métriques produites sont
exploratoires et ne constituent pas des performances finales pour un article.
Les six bases de revue ne contenaient aucun récit `validated` lors de la mise en
place du protocole.

## Préparer les données sur le PC

Depuis la racine de SAFER, après import des annotations directes de
`service_temporaire` :

```powershell
.\text\.venv\Scripts\python.exe text\scripts\sat_lodo.py preflight
.\text\.venv\Scripts\python.exe text\scripts\sat_lodo.py prepare --out annotation\lodo_sat_transfer\sat12l_sm_lodo_v1
.\text\.venv\Scripts\python.exe text\scripts\sat_lodo.py check-bundle --bundle annotation\lodo_sat_transfer\sat12l_sm_lodo_v1
```

Le lot `service_temporaire` a été importé localement le 17 octobre 2026 :
2 523 préannotations directes et 416 réponses rejetées. Le précontrôle retient
980 récits de ce corpus selon les critères ci-dessous. Le dossier préparé est
`annotation/lodo_sat_transfer/sat12l_sm_lodo_v1/`. **Il est ignoré par Git et
contient les récits complets.** Transférez-le vers un répertoire privé du
cluster, hors dépôt Git. Ne commitez que le code sous `text/`.

La préparation garde uniquement les annotations directes v2 aux extraits
exacts, ordonnées et sans chevauchement, sans `issues`, sans unité incertaine,
sans saut de ligne au sein d'une unité et sans trou alphanumérique entre les
unités. Le manifeste donne les effectifs et les exclusions par corpus. Les
passages entre unités, s'ils ne portent aucun caractère alphanumérique, sont
conservés dans une **copie dérivée d'entraînement** en préfixe de l'unité
suivante. Les retours à la ligne de ces passages deviennent des espaces dans
cette seule copie ; les textes et positions originaux restent inchangés dans
les JSONL d'évaluation. Les trois partitions par rotation et les identifiants
de partage 80/20 sont sauvegardés avec leurs empreintes SHA-256.

## Environnement du cluster

Le code arrive sur le cluster par votre `commit`/`push` manuel. Le dossier
**privé** `annotation/lodo_sat_transfer/sat12l_sm_lodo_v1/` reste hors de Git :
copiez-le séparément, de préférence dans
`$SCRATCH/safer_private/sat12l_sm_lodo_v1/`. Le job le détecte aussi à
`SAFER/annotation/lodo_sat_transfer/sat12l_sm_lodo_v1/` si vous préférez ce
chemin. Il doit contenir `manifest.json`, `complete.json` et `folds/`.

Une copie du dépôt officiel `wtpsplit` est incluse dans
`text/vendor/wtpsplit/` et poussée avec le code SAFER. Son trainer LoRA dépend
de plusieurs modules internes au dépôt
[officiel](https://github.com/segment-any-text/wtpsplit/blob/main/README.md#adapt-to-your-own-corpus-via-lora).
La copie correspond à la révision
`3ef928c8e163d11dc6f6336ad610bff2ffe0bda4` ; ses fichiers et leurs
empreintes sont consignés dans `text/vendor/wtpsplit/SAFER_VENDOR.json`.
Les métadonnées Git internes ont été retirées pour que le code soit réellement
versionné par SAFER, et la licence MIT est conservée. Aucun clone GitHub n'est
fait sur le cluster. Le premier job GPU crée un environnement isolé et installe
PyTorch 2.5.1 CUDA 12.4, le fichier
`text/annotation/requirements_sat_lodo.txt`, le dépôt en mode éditable et
`adapters==0.2.1` sans ses dépendances automatiques. Ce runtime et les
versions installées sont conservés dans `${SCRATCH:-$HOME}/safer_sat_lodo/runtime/`.
`wandb` est fixé à `0.15.12` : le trainer l'importe même si `report_to=none`,
et cette version dispose d'une roue Python qui évite la compilation Go du
paquet récent. Aucune journalisation W&B n'est activée.
Cette étape suppose que les nœuds GPU ont accès à PyPI, l'index CUDA
et Hugging Face pour installer les dépendances et obtenir le modèle. Si le cluster bloque ces téléchargements depuis les nœuds
GPU, le journal du premier job indiquera l'étape en échec.

Les sorties vont automatiquement dans
`${SCRATCH:-$HOME}/safer_sat_lodo/runs/sat12l_sm_lodo_v1/`. Aucune variable
d'environnement n'est requise pour les chemins standard. Le job vérifie le
GPU, les versions, les empreintes du bundle et la révision de `wtpsplit` avant
chaque entraînement. PyTorch 2.5.x est retenu car le trainer officiel appelle
`torch.load` sans `weights_only=False`, incompatible avec le défaut de 2.6.

La [configuration officielle d'exemple](https://github.com/segment-any-text/wtpsplit/blob/main/configs/lora/lora_dummy_config.json)
sert de base au fichier `text/annotation/sat_lodo_lora_template.json`. Ici,
`model_name_or_path` vaut `segment-any-text/sat-12l-sm`, le lot GPU vaut 8 avec
accumulation 8, et `subsample` est désactivé. Les autres choix substantiels
(30 époques, taux `3e-4`, LoRA `r=16`, `alpha=32`) restent ceux de l'exemple.
La journalisation externe est désactivée ; les sorties restent sur le cluster.

## Essai GPU puis six tâches

Depuis la racine `SAFER/` du clone sur le cluster, après avoir transféré le
dossier privé, la seule commande de préparation est :

```bash
sbatch text/jobs/sat_lodo_smoke.sh
```

Ce job installe les dépendances, entraîne un seul pas, recharge l'adaptateur
avec `SaT("sat-12l-sm", lora_path=...)` et vérifie un découpage. Attendez sa
fin **réussie** ; le fichier
`${SCRATCH:-$HOME}/safer_sat_lodo/runs/sat12l_sm_lodo_v1/smoke_ok.json`
est alors créé. Ensuite, toujours depuis `SAFER/` :

```bash
sbatch text/jobs/sat_lodo_gpu.sh
```

Le tableau `0-5%2` et toutes les ressources Slurm sont déjà écrits dans le
script. Si vous êtes dans `SAFER/text/`, utilisez les mêmes commandes sans
le préfixe `text/`. Les journaux `slurm-sat-lodo*` sont écrits dans le
répertoire depuis lequel vous avez lancé `sbatch`.

Une tâche relancée reprend le dernier checkpoint officiel disponible. Si son
adaptateur final existe déjà, elle reprend directement l'évaluation. Chaque
tâche écrit `target_<corpus>/metrics.json`, les courbes de seuils sur la
validation source et une base SQLite de prédictions permettant de reprendre
l'évaluation sans recommencer les inférences terminées.

Les valeurs de loss d'entraînement et de validation sont aussi conservées dans
`target_<corpus>/trainer/trainer_state.json` (historique complet du trainer)
et dans `target_<corpus>/loss_history.csv` (une ligne par valeur journalisée,
avec le pas et l'époque). Le fichier CSV est produit à la fin de
l'entraînement ; pendant l'exécution, les valeurs restent visibles dans le
journal Slurm `.out`. Le job smoke produit ses propres fichiers analogues sous
`smoke/`. Les checkpoints intermédiaires conservent l'historique atteint au
moment de chaque sauvegarde, ce qui aide en cas d'interruption.

Une fois les six tâches terminées :

```bash
sbatch text/jobs/sat_lodo_summary.sh
```

`summary.csv` présente M0, M1, M3 et M3−M1 par corpus, puis leur moyenne
macro. Les seuils de M1 et M3 sont sélectionnés **séparément dans chaque
rotation** sur les cinq sources, jamais sur la cible. Le seuil `0,10` de la
calibration antérieure BTP–métallurgie n'est pas réutilisé. Les métriques
principales sont le F1 de paires `(start,end)` exactement égales ; les rapports
contiennent aussi un F1 des frontières brutes et le rappel exact par rôle.

Les fichiers du bundle et des résultats contiennent des récits d'accidents :
conservez-les dans un espace privé du cluster. Cette expérience ne remplace pas
une évaluation ultérieure sur un gold humain indépendant.
