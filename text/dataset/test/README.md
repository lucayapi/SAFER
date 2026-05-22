# Corpus de test (hors BTP)

Évaluation **out-of-domain** : récits hors domaine BTP, même schéma que `dataset/data_btp.csv`.

## Registre central

Les corpus disponibles sont déclarés dans [`configs/test_corpora.yaml`](../../configs/test_corpora.yaml).

| Champ | Description |
|-------|-------------|
| `id` | Clé courte (`metallurgie`, `chimie`, …) |
| `data_csv` | Métadonnées + `sentence`, `pred_label`, `pred_ok`, `accident_id` |
| `emb_csv` | Embeddings Qwen figés (SCGM strict fidelity) |

**Convention** pour ajouter un corpus :

1. Placer `dataset/test/data_<id>.csv`
2. Générer `embeddings/test/Qwen3-Embedding-0.6B_<id>.csv` :
   ```bash
   python scripts/export_test_embeddings.py --corpus <id>
   ```
3. Ajouter l’entrée dans `configs/test_corpora.yaml`
4. Lancer les pipelines avec `TEST_CORPUS=<id>` ou `--corpus <id>`

## Corpus fourni

| Fichier | Rôle |
|---------|------|
| `data_metallurgie.csv` | Métallurgie (défaut du registre) |

## Utilisation

```bash
# Variable d'environnement (jobs SLURM / bash)
export TEST_CORPUS=metallurgie

# CLI
python scripts/run_tpn_macro_transfer_discovery.py --base-method scgm_text --corpus metallurgie
sbatch jobs/train_scgm_text.sh   # avec TEST_CORPUS dans l'environnement
sbatch jobs/export_raw_geometry.sh
sbatch jobs/export_test_embeddings.sh
CORPUS=metallurgie bash jobs/run_tpn_macro_transfer.sh
```

**Sorties** (sous `output_test/<corpus_id>/`) :

- SCGM / contrastifs : `<method>/metrics/metrics_geometry_test.csv`
- Raw embedding : `raw_embedding/metrics/metrics_geometry.csv`
- Transfert macro + topics : `macro_transfer/<method>/` (BERTopic + `theme_label` via `bertopic.representation` par défaut)
- BN (notebook 04) : `bn_staging/`

Voir aussi la section **Arborescence** dans [`README.md`](../../README.md).

Les métriques de test utilisent les **best models** entraînés sur BTP :

- **Contrastifs** : encodeur fine-tuné → embeddings du corpus test
- **SCGM** (strict fidelity) : embeddings Qwen figés + checkpoint SCGM
- **Embedding brut** : `export_raw_embeddings.py --config configs/methods/raw_embedding_test.yaml` (ajuster `test_corpus` dans le YAML)
