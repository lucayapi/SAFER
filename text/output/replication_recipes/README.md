# Réplications multi-seeds

`replication_config.yaml` est la configuration unique de la campagne. Modifier
`training.n_seeds` change seulement le nombre de réplications; les seeds sont
générées par `training.seed_generator` et sont identiques pour les trois
méthodes. Conserver `training.split_seed: 42` afin de garder exactement les
mêmes folds BTP groupés par `accident_id`.

Depuis `text/` sur le Mésocentre :

```bash
bash jobs/submit_replications.sh
```

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
