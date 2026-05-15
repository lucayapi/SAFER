# Notebooks SCGM / MALT

## 01 — SCGM texte BTP

`01_scgm_text_btp_experiment.ipynb` : exploration, entraînement SCGM-Text, export, évaluation et figures pour le sous-corpus BTP.

## 02 — MALT BTP → Métallurgie

`02_malt_btp_to_mettalurgie_transfer.ipynb` : transfert macro-ancré vers la métallurgie avec un K global.

Prérequis :

- checkpoint source `runs/scgm_text_qwen06/best_model.pt`
- `dataset/data_metallurgie.csv`
- embeddings cible Qwen 0.6B dans `embeddings/`

Exécution Papermill (depuis `text/`) :

```bash
cd text
papermill notebooks/02_malt_btp_to_mettalurgie_transfer.ipynb \
  runs/malt_btp_to_mettalurgie_qwen06/executed_notebook.ipynb \
  -p SOURCE_CHECKPOINT runs/scgm_text_qwen06/best_model.pt \
  -p TARGET_DATA_CSV dataset/data_metallurgie.csv \
  -p TARGET_EMB_CSV embeddings/Qwen3-Embedding-0.6B_metallurgie.csv \
  -p OUTPUT_DIR runs/malt_btp_to_mettalurgie_qwen06 \
  -p RUN_TRAINING true \
  -p N_SUBCLASS 32
```
