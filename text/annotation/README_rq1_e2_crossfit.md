# Campagne RQ1 E2 avec caractéristiques SaT+LoRA hors pli

Cette campagne est isolée des anciens résultats. Elle utilise le GOLD `narrative_v3` et les partitions figées dans le paquet RQ1. Pour chaque rotation, A est le `train` source 80 %, V le `val` source 20 %, et le test cible est repris tel quel.

## Protocole

- A est réparti en cinq plis déterministes, équilibrés par corpus source. Les récits dont le texte devient identique après normalisation des espaces et de la casse restent dans le même pli, même s'ils apparaissent dans plusieurs corpus. Les identifiants des plis et leur empreinte sont inscrits dans `split_manifest.json`.
- Chaque modèle de pli est sélectionné sur une validation interne de 10 % extraite des quatre autres plis de A. Checkpoint, époque et seuil sont choisis sur cette validation uniquement. Le modèle est ensuite réentraîné sur les quatre plis permis pour produire les caractéristiques du pli tenu à l'écart.
- Un M3 supplémentaire est sélectionné en validation interne puis réentraîné sur tout A. Il produit les caractéristiques de V et du test cible. Les poids de validation/test ne participent pas au choix des E2.
- E2-A utilise le score de frontière SaT; E2-B les deux états contextuels projetés; E2-C les états plus le score SaT; E2-D ajoute la position relative. Projection fixe : 128 dimensions, seed `20261018`.
- Les seules frontières négatives supervisées sont les coupures candidates strictement à l'intérieur d'une unité GOLD. Les frontières dans du texte exclu ou non annoté restent inconnues et sont masquées dans la perte.
- Les seuils de candidats `0.02/0.05/0.10`, les pénalités du décodeur et les checkpoints E2 sont sélectionnés sur V. Les tests cibles ne servent qu'au rapport final.
- E0 reprend le seuil interne choisi pour le M3 final; E1 recalibre le seuil du même M3 sur V. Les variantes E2 utilisent trois graines fixes. Les sorties comprennent F1 exact, Soft Span F1, Soft Span F1 IoU≥0,75, Boundary F1, fusions/fragmentations et IC bootstrap appariés.

Les caractères de frontière sont extraits avec la fenêtre SaT officielle et rapprochés des probabilités produites par le même adaptateur. Le pilote réalise aussi un entraînement LoRA réduit, recharge l'adaptateur et contrôle les offsets.

## Cluster

Copier/committer d'abord le code versionné, et placer le paquet privé au chemin déjà reconnu par `text/jobs/_rq1_final_env.sh`. Depuis `SAFER/`, lancer le pilote BTP :

```bash
sbatch text/jobs/rq1_e2_crossfit_launch.sh
```

Le lanceur soumet le précontrôle, le smoke GPU, cinq plis BTP (au plus deux GPU), le M3 final BTP, l'entraînement E2 BTP et un rapport pilote. Les fichiers Slurm `slurm-rq1-e2x-*.out/.err` consignent chaque étape. Les résultats privés sont sous `$SCRATCH/safer_sat_rq1_refiner_crossfit/runs/rq1_e2_crossfit_v1` (ou `$HOME` si `SCRATCH` n'est pas défini).

Après examen de `PILOT_RESULTS.md`, des CSV, de la courbe SVG et des logs, le chercheur peut explicitement lancer les cinq autres rotations :

```bash
sbatch text/jobs/rq1_e2_crossfit_launch.sh --remaining
```

Ce mode vérifie la présence du marqueur de fin du pilote BTP avant de soumettre les plis restants. Il ne choisit aucun paramètre sur le test BTP. Les sorties finales n'apparaissent qu'après réussite de toutes les rotations.

Une reprise du même job vérifie les marqueurs et empreintes avant de réutiliser les adaptateurs ou caches existants. Ne pas modifier le paquet GOLD, les scripts ou le registre des corpus au milieu de cette campagne : une modification de code/GOLD impose une nouvelle version isolée.

## Portée de l'inférence

Le cross-fitting retire l'exposition directe de M3 aux récits utilisés pour ajuster E2. Il ne constitue pas une confirmation totalement indépendante, car l'architecture E2 a déjà été choisie après consultation de tests exploratoires. Le rapport le précise.
