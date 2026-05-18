# Corpus de test (hors BTP)

Évaluation **out-of-domain** : récits métallurgie, même schéma que `dataset/data_btp.csv`.

| Fichier | Rôle |
|---------|------|
| `data_metallurgie.csv` | Métadonnées + `sentence`, `pred_label`, `pred_ok`, `accident_id` |

Les métriques de test (`metrics_geometry_test.csv`) utilisent les **best models** entraînés sur BTP :

- **Contrastifs** : encodeur fine-tuné → embeddings du corpus test
- **SCGM** (strict fidelity) : embeddings Qwen figés (`embeddings/test/Qwen3-Embedding-0.6B_metallurgie.csv`) + checkpoint SCGM
- **Embedding brut** (référence) : `resultats/raw_embedding_test/metrics/metrics_geometry.csv` (postprocess ou `export_raw_embeddings.py` + `configs/methods/raw_embedding_test.yaml`)
