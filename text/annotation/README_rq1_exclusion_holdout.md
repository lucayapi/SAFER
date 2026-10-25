# Raffinement contextuel des faits et passages exclus : campagne allégée

Une seule graine (13) et six rotations leave-one-domain-out. Dans chaque
rotation, les récits d'entraînement des cinq corpus sources sont partagés par
récit, avec regroupement des textes identiques : environ 80 % entraînent un
SaT auxiliaire ; les 20 % restants fournissent des prédictions hors échantillon
et les annotations qui entraînent le raffineur. Le GOLD v3 actuel contient
1 727 à 2 407 récits et 292 à 702 passages explicitement exclus dans cette
part selon la cible. Un SaT final est ensuite ajusté sur **100 %** de
l'entraînement source et fournit les représentations de la validation source
et du test cible. Aucun récit cible ne sert à l'entraînement ou à la sélection.

Le raffineur utilise les représentations contextuelles à gauche et à droite
des coupures candidates. Les bords des faits et des passages explicitement
exclus sont positifs, leurs intérieurs négatifs, et les zones non annotées
restent masquées. L'époque et les seuils sont choisis sur la validation source
par la moyenne harmonique du F1 exact des faits et du rappel exact des
exclusions ; les deux mesures sont ensuite rapportées séparément sur chaque
test cible. Cette campagne évalue la **délimitation**, pas le rejet automatique
des passages exclus ni la classification des rôles.

Le partage disjoint réduit le coût par rapport au cross-fitting à cinq plis :
par rotation, il faut un SaT auxiliaire, un SaT final et un raffineur. Chaque
SaT comprend une sélection interne puis un réajustement sur toutes ses données
autorisées. Le lanceur réunit ces étapes en **six tâches GPU** (une par corpus
cible), après un contrôle GPU initial, au lieu de 42 tâches GPU d'entraînement
dans le lanceur à cinq plis. Chaque tâche est donc plus longue. Trois tâches
GPU au maximum peuvent tourner en parallèle.
L'inconvénient est que le raffineur ne reçoit des exemples annotés que sur
20 % de l'entraînement source ; il faut le signaler dans l'article et vérifier
les résultats par domaine. Les prédictions du SaT final sur les mêmes tests
sont sauvegardées pour une comparaison appariée ultérieure, sans entraîner
un autre modèle.

Depuis la racine `SAFER`, après transfert du code et du paquet GOLD privé :

```bash
sbatch text/jobs/rq1_exclusion_holdout_launch.sh
```

Le cluster doit disposer du paquet GOLD RQ1 complet (manifest, complete,
rq1_runtime et folds), du dépôt à jour dont `text/vendor/wtpsplit`, d'une GPU
A100 ou H100, des modules `gcc/8.1.0` et `python/3.10.10`, ainsi que des
modèles SaT et du tokenizer dans le cache ou accessibles au téléchargement.
Le job de préparation crée ou vérifie l'environnement dédié `.venv1`. Vérifier
que `SCRATCH` est partagé et offre une marge suffisante pour trois tâches GPU
de 96 Go de mémoire hôte chacune et leurs fichiers temporaires.

Les résultats se trouvent par défaut dans
`$SCRATCH/safer_sat_rq1_exclusion_holdout/runs/seed13_v1/`. Les tableaux
`summary/per_domain.csv` et `summary/macro.csv` résument cette seule méthode.
Chaque dossier `target_<corpus>/predictions/` contient les sorties du SaT
final et du raffineur sur les mêmes récits tests. Un seul lancement ne mesure
pas la variabilité entre graines.

Le lanceur cross-fit précédent (`rq1_exclusion_boundaries_launch.sh`) reste
inchangé et n'est pas annulé automatiquement. Modifier ou lancer le nouveau
script ne change pas les jobs Slurm déjà soumis.
