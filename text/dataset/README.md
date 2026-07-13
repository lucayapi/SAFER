# Données annotées (local uniquement)

Les CSV `data_<corpus>.csv` ne sont pas versionnés (fichiers lourds).

Génération :
1. Annoter via `annotation/` → `annotation/outputs/<run_id>/`
2. Migrer : `python scripts/migrate_annotation_to_dataset.py`
3. Encoder : `sbatch jobs/export_corpus_embeddings.sh`

Corpus attendus : `data_btp.csv`, `data_metallurgie.csv`, `data_caou.csv` (registre `configs/test_corpora.yaml`).
