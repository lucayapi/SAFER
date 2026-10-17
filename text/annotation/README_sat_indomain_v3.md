# SaT + LoRA : diagnostic dans chaque domaine et transfert depuis une seule source

Cette expérience complète RQ1 sans remplacer les six rotations multisources LODO.
Elle utilise les mêmes 17 927 récits `narrative_v3` confirmés, le modèle
`sat-12l-sm`, le trainer LoRA déjà épinglé et la même définition des scores
(segments exacts et frontières après neutralisation des espaces périphériques).
La référence est le jeu d'annotations confirmé par la chercheuse ; les résultats
restent conditionnés à sa qualité et à sa convention d'annotation.

## Plan d'expérience

Pour chacun des six corpus, `prepare` fait un partage déterministe 70/15/15
**par récit**. Les récits ayant un texte original identique dans un corpus
restent dans la même partition. Les identifiants, empreintes SHA-256, effectifs
et motifs d'exclusion de l'entraînement sont conservés dans le paquet privé.
Les récits de validation et de test conservent leurs annotations originales,
y compris ceux qui ne sont pas compatibles avec le format d'entraînement.

Chacune des six tâches GPU entraîne uniquement sur les 70 % de **son propre
corpus**. Les checkpoints et le seuil de SaT + LoRA (M2) sont choisis ensemble
sur ses 15 % de validation selon le F1 micro des segments exacts. Le seuil de
SaT initial calibré (M1) est choisi sur cette **même validation** ; M0 utilise
son seuil par défaut. En cas d'égalité, le seuil le plus proche du défaut,
puis le checkpoint le plus ancien sont préférés. Le pas de sauvegarde est de
200, sauf pour le plus petit corpus (chimie–plastiques : 50) afin d'avoir au
moins un checkpoint intermédiaire. Tous les autres hyperparamètres proviennent
de la même configuration officielle épinglée.

Après la sélection, chaque modèle M2 est évalué sur les six ensembles de test
de 15 %, identiques pour toutes les lignes de la matrice. La diagonale est le
diagnostic **dans le domaine**. Les cases hors diagonale sont un transfert
**single-source** sans adaptation à la cible. Le seuil d'une ligne est toujours
celui choisi sur la validation de sa source ; aucun test cible ne sert à le
recalibrer. Cette matrice ne remplace pas l'expérience LODO multisource.

La préparation de l'entraînement est celle du trainer officiel : les extraits
factuels littéraux sont livrés comme liste d'unités, puis regroupés par celui-ci.
Les passages exclus entre unités ne font donc pas partie de cette copie. Les
tests, eux, portent sur les récits complets originaux. Si la diagonale est
faible, examiner aussi ce décalage de représentation avant de conclure à une
incapacité intrinsèque du modèle ou à un défaut des annotations.

## Lancement Slurm

Depuis `SAFER/` sur le cluster, après avoir synchronisé les fichiers de code,
conservé le paquet privé `sat12l_sm_gold_v3_rq1` de l'expérience précédente et
installé l'environnement `.venv1` déjà utilisé par les jobs RQ1 :

```bash
prep=$(sbatch --parsable text/jobs/sat_indomain_v3_prepare.sh)
train=$(sbatch --parsable --dependency=afterok:$prep text/jobs/sat_indomain_v3_gpu.sh)
sbatch --dependency=afterok:$train text/jobs/sat_indomain_v3_summary.sh
```

Le premier job reconstruit le nouveau paquet à partir de la référence privée
déjà présente sur le cluster. Il n'envoie aucune donnée dans Git. Le deuxième
est un tableau Slurm `0-5%2` avec au plus deux GPU simultanés. Le troisième ne
s'exécute que si **toutes** les tâches GPU ont réussi. Les journaux Slurm
`slurm-sat-indomain-*.out/.err` se trouvent dans le dossier de soumission.

Par défaut, le paquet privé est dans
`${SCRATCH:-$HOME}/safer_sat_indomain/bundle/sat12l_sm_gold_v3_indomain_v1`
et les résultats dans
`${SCRATCH:-$HOME}/safer_sat_indomain/runs/sat12l_sm_gold_v3_indomain_v1`.
Chaque `source_<corpus>/` contient la configuration, les pertes, les
checkpoints, le choix sur validation, les caches de probabilités, les scores
par récit et `metrics.json`. À la racine des résultats :

- `indomain_summary.csv` : M0/M1/M2 et gain M2−M1 sur la diagonale ;
- `transfer_exact_f1.csv` et `transfer_exact_f1.svg` : matrice 6 × 6 ;
- `transfer_metrics_long.csv` : précision, rappel et F1 exact/frontières,
  fusions et fragmentation pour chaque source, cible et modèle ;
- `summary_manifest.json` : empreintes des résultats et des choix de modèles.

Une tâche interrompue peut être soumise de nouveau avec le même script ; le
trainer reprend du dernier checkpoint complet et les caches de probabilités
reprennent par lots fixes. Ne modifiez ni le code ni les données d'un run en
cours : sa signature les contrôle. Pour vérifier un paquet préparé avant un
lancement GPU, exécuter `check-bundle` dans l'environnement `.venv1`.
