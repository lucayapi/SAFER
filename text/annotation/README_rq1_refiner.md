# Raffinement factuel guidé par SaT — première campagne exploratoire

Depuis la racine de `SAFER` sur le cluster :

```bash
sbatch text/jobs/rq1_refiner_launch.sh
```

Le lanceur soumet successivement un précontrôle, un essai GPU réduit, six
rotations LODO en tableau `0-5%2` et une synthèse après réussite des six
rotations. Chaque tâche du tableau utilise un GPU. Les progrès figurent dans
`slurm-rq1-refiner-*.out` et `.err` dans le dossier de soumission.

Le job réutilise le GOLD figé de RQ1, les six adaptateurs M3 sélectionnés et
l'environnement `.venv1` de SaT ; le précontrôle le prépare s'il manque.
Par défaut, il cherche les modèles dans
`${SCRATCH:-$HOME}/safer_sat_rq1_final/runs/rq1_segmentation_final_v1`, puis
dans `SAFER/rq1_segmentation_final_v1`. Il n'écrase aucun résultat RQ1.
Les sorties privées sont placées dans
`${SCRATCH:-$HOME}/safer_sat_rq1_refiner/runs/rq1_refiner_exploratory_v1`.
Si le cluster utilise d'autres emplacements, seules les variables
`RQ1_REFINER_M3_ROOT` et `RQ1_REFINER_OUT` sont à définir avant `sbatch`.

Chaque rotation conserve un cache reprenable des états contextuels projetés
et des probabilités de M3, trois meilleurs réseaux par variante, leurs pertes,
sélections sur validation source, prédictions avec offsets et métriques.
`summary.csv`, `summary_by_method.csv`, `summary.json` et `RESULTS.md` ne sont
produits que si les six rotations et
tous les tests attendus sont présents. E0 est M3 tel que sélectionné par RQ1 ;
E1 réévalue son seul seuil sur les validations sources ; E2 filtre les
frontières ; E3 sélectionne des segments par décodage global ; E4 ajoute les
pertes fusion/fragmentation. Les seuils candidats sont 0,02, 0,05 et 0,10.
E1 est limité à cette grille permissive : ce n'est pas une recherche de seuil
plus large que celle déjà faite pour M3. Les états cachés de M3 sont projetés
par une matrice orthogonale fixe à 128 dimensions pour limiter le stockage ;
la graine et la dimension sont consignées dans le code et le manifeste.
Relancer la même commande après une interruption est sans danger : les caches
des récits et les meilleurs petits réseaux déjà terminés sont réutilisés.
Une dépendance devenue impossible annule les tâches aval du lanceur ; il faut
alors lire le `.err` du premier job en échec et relancer la commande après
correction.

**Limite méthodologique :** M3 a déjà vu ses récits sources d'entraînement.
Les caractéristiques utilisées pour entraîner E2–E4 sur ces récits sont donc
en échantillon pour M3. Les tests cibles restent séparés, mais cette campagne
est exploratoire. Pour une analyse définitive de l'article, il faudra
recalculer des caractéristiques M3 hors échantillon par plis internes. Les
validations sources ont également servi auparavant à sélectionner M3 ; elles
servent ici à régler le raffineur. Aucun réglage ne repose sur les tests cibles.
