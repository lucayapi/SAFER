# Jeux textuels GOLD narrative_v3

Les six `data_<corpus>.csv` proviennent exclusivement de
`annotation/rq1_sat_v3_20261017/gold_snapshot.jsonl` (SHA256
`4845dbec927c5cd1161360729382ac5497c3f5c41d1562b03d5020c2f20bdca8`).
Le snapshot compte 17 927 récits, dont certains sans unité annotée, et 147 487
unités. Chaque ligne CSV est une unité, avec le récit complet, l'extrait exact,
les positions Python `[start,end)`, le rôle et la provenance de validation.
`pred_label` est une colonne de compatibilité ; ce n'est plus une prédiction LLM
à recalculer. Les classes sont `A0`, `A1`, `B`, `C` et `AUTRE`.

Depuis la racine SAFER, pour régénérer et vérifier les fichiers privés :

```bash
python text/scripts/build_gold_v3_datasets.py build
python text/scripts/build_gold_v3_datasets.py check
```

Le manifeste `gold_v3_manifest.json` conserve les empreintes et effectifs. Les
quatre CSV antérieurs sont sauvegardés dans `legacy_pre_gold_v3_20261018/`.
Les CSV et l'archive restent ignorés par Git. Copier les six CSV et le manifeste
vers `SAFER/text/dataset/` sur le cluster avant tout export d'embeddings.

Depuis `SAFER` sur le cluster :

```bash
sbatch --array=0-17%2 text/jobs/export_gold_embeddings.sh
sbatch --export=ALL,BACKBONE=qwen3,CORPUS=btp text/jobs/export_gold_embeddings.sh
```

Le premier lance les 18 couples corpus–encodeur (au plus deux GPU à la fois).
Les valeurs de `BACKBONE` sont `qwen3`, `e5`, `camembert`. Les valeurs de
`CORPUS` sont `btp`, `metallurgie`, `caou`, `bois_textille`,
`service_temporaire`, `nicollin`. Les vecteurs vont dans
`text/embeddings/gold_v3/<backbone>/<corpus>.csv`, avec un manifeste adjacent.
La reprise vérifie l'empreinte de la source et des vecteurs avant d'ignorer un
export. Un export d'une autre version est refusé. Aucun entraînement de
classification n'est lancé par ce job.
