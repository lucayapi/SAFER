# Réplications multi-seeds

`replication_config.yaml` est la configuration unique de la campagne. Modifier
`training.n_seeds` change seulement le nombre de réplications; les seeds sont
générées par `training.seed_generator` et sont identiques pour les trois
méthodes. Conserver `training.split_seed: 42` afin de garder exactement les
mêmes folds BTP groupés par `accident_id`.

Depuis `text/` sur le Mésocentre :

```bash
# Une seule méthode à la fois : à lancer puis attendre sa fin avant la suivante.
bash jobs/submit_replications_softtriple.sh
bash jobs/submit_replications_supcon.sh
bash jobs/submit_replications_cross_entropy.sh
```

Chaque seed efface automatiquement ses checkpoints, caches, embeddings et
artefacts de folds après avoir validé ses métriques et prédictions. Les CSV
nécessaires au bootstrap restent présents. `storage.max_parallel_seeds: 1`
limite aussi le job array à une seule seed à la fois; l'augmenter utilise plus
de GPU et plus d'espace temporaire.

Le script crée `output/replications/<modèle>/seed_<seed>/`. Rapatrier de chaque
dossier les fichiers `run_manifest.json`, `configs/config_resolved.yaml`,
`cv/`, `metrics/` et `predictions/`. Les checkpoints peuvent rester sur le
Mésocentre.

Après rapatriement, depuis `text/` localement :

```bash
python scripts/analyze_replications.py
```

Les résultats sont écrits dans `output/replication_analysis/`. Le bootstrap
rééchantillonne des accidents complets, 2 000 fois par défaut; ce nombre se
modifie par `bootstrap.n_resamples`.
