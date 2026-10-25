# Adaptation de SaT + LoRA aux unités élémentaires

Une unité élémentaire est ici un intervalle explicitement annoté : soit une
unité factuelle `units`, soit un passage `excluded_spans`. Le script
`text/scripts/sat_elementary_lodo.py` trie ces intervalles selon leurs offsets
et les fournit comme segments littéraux au même entraîneur SaT + LoRA que le
M3 historique. Les lacunes non annotées ne deviennent pas des exemples de
rejet. Ce modèle apprend des **frontières**, pas le statut factuel ni les rôles.

Les six rotations reprennent exactement les partitions du paquet RQ1
`sat12l_sm_gold_v3_rq1` : cinq corpus sources avec 80 % d'entraînement et
20 % de validation par source, puis le sixième corpus entier en test. Une seule
graine est utilisée. L'architecture, les hyperparamètres LoRA et les révisions
du modèle sont inchangés. Le seuil est choisi sur la validation source par le
F1 exact des **unités élémentaires** (faits et exclus réunis), moyenné sur les
cinq corpus sources. Le tableau final donne ce F1 et les rappels exacts des
faits et des exclus sur chaque corpus cible. Un exclus correctement isolé est
ainsi une bonne prédiction de segmentation. Aucun résultat de test ne sert à
sélectionner le seuil. Puisque le critère de sélection diffère de celui du M3
historique, leur comparaison brute n'isole pas à elle seule l'effet de l'ajout
des passages exclus aux exemples d'entraînement.

## Préparer les données locales

Depuis la racine `SAFER/` :

```powershell
.\text\.venv\Scripts\python.exe text\scripts\sat_elementary_lodo.py prepare
.\text\.venv\Scripts\python.exe text\scripts\sat_elementary_lodo.py check
```

Le paquet créé est `annotation/lodo_sat_transfer/sat12l_sm_elementary_v1/`.
Il est ignoré par Git. Il contient six fichiers `training.pth`, un manifeste
avec les empreintes et le nombre de faits et d'exclusions utilisés. Les récits
GOLD et le modèle SaT initial restent dans le paquet RQ1 existant ; il faut
donc **les deux dossiers** sur le cluster. Copier le nouveau paquet vers
`$SCRATCH/safer_private/sat12l_sm_elementary_v1/` ou garder sa place sous
`SAFER/annotation/lodo_sat_transfer/`. Le paquet RQ1 reste à son emplacement
habituel. Aucun texte privé ne doit être ajouté à Git.

## Soumettre une campagne à une seule graine

Depuis la racine `SAFER/` sur le cluster :

```bash
smoke=$(sbatch --parsable text/jobs/sat_elementary_smoke.sh)
train=$(sbatch --parsable --dependency=afterok:${smoke} text/jobs/sat_elementary_gpu.sh)
sbatch --dependency=afterok:${train} text/jobs/sat_elementary_summary.sh
```

Cela soumet un smoke GPU, six tâches GPU LODO avec au plus trois simultanées,
puis un résumé CPU après succès des six. Le smoke vérifie qu'un adaptateur
entraîné pendant un pas peut être rechargé. Les résultats vont dans
`$SCRATCH/safer_sat_lodo/runs/sat12l_sm_elementary_v1/` : un sous-dossier
`target_<corpus>/` par rotation, avec l'adaptateur, les seuils de validation,
les prédictions et `metrics.json`, puis `summary.csv` à la racine.

Le chemin du nouveau paquet peut être surchargé par `SAT_ELEMENTARY_PREPARED`.
Le paquet RQ1 utilise `SAT_LODO_BUNDLE`. Si `$SCRATCH` est absent, les scripts
utilisent `$HOME`. L'entraînement complet nécessite les GPU du cluster ; la
préparation et les tests de format peuvent être effectués localement.
