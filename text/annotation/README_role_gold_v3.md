# Classification des rôles sur les unités GOLD `narrative_v3`

Cette campagne est distincte de la segmentation RQ1 et des anciens classeurs
BTP. Elle classe les unités GOLD **déjà délimitées**, avec cinq sorties
`A0/A1/B/C/AUTRE`. Les partitions par récit 70/15/15 et l'empreinte du GOLD
sont héritées du paquet privé RQ1. Les tests sont identiques pour les quatre
méthodes et les trois graines.

## Données requises sur le cluster

Depuis `SAFER/`, conserver les six `text/dataset/data_<corpus>.csv` GOLD et
`text/dataset/gold_v3_manifest.json` (alias `data_caou.csv`). Ces CSV et le
paquet RQ1 sont privés et ignorés par Git. Le paquet doit contenir
`manifest.json` et les six dossiers `folds/target_<corpus>/split_ids.json`.
Le programme cherche dans `annotation/rq1_final_v1/bundle/` et
`annotation/lodo_sat_transfer/rq1_final_v1/`; si nécessaire, fixer
`ROLE_GOLD_BUNDLE` au chemin du dossier qui contient `manifest.json`.

Les lanceurs chargent `text/.venv1` s'il existe, sinon `text/.venv`.
`ROLE_GOLD_VENV` permet de choisir explicitement un autre environnement.
Il doit fournir PyTorch avec CUDA, Transformers, pandas, scikit-learn,
joblib et les dépendances déjà utilisées par `text/contrastive_methods`.

## Lancement depuis `SAFER/`

```bash
sbatch text/jobs/role_gold_single_source.sh
sbatch text/jobs/role_gold_lodo.sh
```

Chaque commande soumet un précontrôle, l'export E5 manquant (ou sa vérification
par empreinte), un essai GPU réduit pour les trois méthodes, six baselines
gelées, un tableau de 54 fine-tunings `0-53%2`, puis le rapport après réussite
de toutes les tâches. Les deux commandes peuvent donc occuper quatre GPU au
total. Les sorties `slurm-role-gold-*` sont dans le dossier depuis lequel
`sbatch` est lancé. Les tâches terminées sont vérifiées et ignorées lors
d'une relance ; un échec arrête la chaîne avant le rapport.

La sortie privée est `safer_role_gold_v3/{single_source,lodo}/` sous `$SCRATCH`
si défini, sinon sous `SAFER/`. `ROLE_GOLD_RUN_ROOT` peut fixer sa racine.
Le précontrôle réserve environ 170 Gio pour 54 entraînements encore à faire ;
il additionne les tâches restantes de l'autre campagne lorsqu'elle est active.
La valeur de base peut être ajustée avec `ROLE_GOLD_MIN_FREE_GIB` après
vérification de la capacité. Il contrôle aussi l'espace des exports E5.
Un meilleur checkpoint complet par entraînement est conservé, sans états
d'optimiseur ni checkpoints intermédiaires.

## Méthode et résultats

- Baseline : vecteurs E5-large gelés, avec préfixe `query: `, régression
  logistique équilibrée et régularisation `C ∈ {0.01, 0.1, 1, 10}` choisie
  sur la validation source. Les exports E5 portent un manifeste et sont
  contrôlés avant toute fusion avec les CSV GOLD.
- CE : E5-large adapté intégralement, tête à cinq sorties, perte CE pondérée.
- SupCon/SoftTriple : E5-large adapté intégralement, projecteur linéaire 128D,
  puis régression logistique ajustée sur les unités d'entraînement source.
  À chaque époque, le couple checkpoint–`C` est choisi par Balanced Accuracy
  sur les seules validations sources. Les trois méthodes utilisent au plus
  15 époques et une patience de trois.

Les fichiers `results/scores.csv`, `scores_by_transfer.csv`,
`recall_by_role.csv` et `confusion_matrices.csv` contiennent les résultats.
La campagne single-source ajoute `matrix_<method>_ba.csv` et
`macro_by_source.csv` ; LODO ajoute `macro_lodo.csv`. Les prédictions par unité
et les modèles sélectionnés restent sous `frozen/` et `trained/`.

La moyenne entre domaines donne le même poids à chaque corpus. Les différences
single-source–multisource ne mesurent pas uniquement l'effet de la diversité :
les volumes d'entraînement diffèrent aussi. Aucun choix de modèle ou de
paramètre n'utilise les étiquettes des domaines cibles.
