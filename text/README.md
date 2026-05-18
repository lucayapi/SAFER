# Pipeline SAFER — texte

Analyse de récits d'accidents : SCGM-G sur embeddings BTP, transfert macro-ancré (MALT), méthodes contrastives, comparaison de topics et réseaux bayésiens exploratoires.

## Installation

```bash
cd text
python3 -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -U pip
bash scripts/install_text_venv.sh
```

Le script utilise `requirements.txt` + **`constraints.txt`** pour figer **numpy 1.26** et **transformers 4.51.3** (Qwen3) et éviter les downgrades vers 4.46.x.

Sur GPU HPC (CUDA 12.4) :

```bash
INSTALL_TORCH_CUDA=1 bash scripts/install_text_venv.sh
```

Installation manuelle : `pip install -r requirements.txt -c constraints.txt`

**Qwen3-Embedding** exige **transformers ≥ 4.51**. Ne pas faire seulement `pip install datasets` ou `pip install -U transformers` sans `-c constraints.txt` (risque numpy 2.x ou retour à transformers 4.46).

Variables d'environnement : `HF_TOKEN` ou `HUGGING_FACE_HUB_TOKEN` dans `.env` (modèles Hugging Face). `OPENAI_API_KEY` optionnel (enrichissement de thèmes). Ne jamais committer `.env`.

## Organisation

| Dossier | Rôle |
|---------|------|
| `dataset/` | CSV métadonnées BTP |
| `dataset/test/` | Corpus test hors domaine (`data_metallurgie.csv`) |
| `embeddings/` | Embeddings pré-calculés (local, gitignored) |
| `configs/` | `paths.yaml`, `methods/*.yaml`, configs SCGM/MALT |
| `safer_core/` | Chemins centralisés → `resultats/` |
| `scgm_text/` | Modèle et entraînement SCGM-G texte |
| `malt_text/` | Transfert MALT-EM |
| `contrastive_methods/` | Batch Triplet, SoftTriple, SupCon |
| `learn_embeddings/` | Export embeddings bruts (encodeur figé) |
| `topic_eval/` | Comparaison MALT / BERTopic / KMeans |
| `bn_malt/` | Réseaux bayésiens (pgmpy) |
| `scripts/` | CLI entraînement, export, évaluation, agrégation |
| `jobs/` | Scripts SLURM Mésocentre |
| `notebooks/` | Analyse (**.ipynb gitignored**, régénération locale via `scripts/build_*.py`) |
| `legacy/` | Code historique (anciens jobs, hors contrastif) |
| `resultats/` | **Toutes les sorties** (gitignored) |

## Sorties (`resultats/`)

| Chemin | Contenu |
|--------|---------|
| `resultats/scgm_text/` | `checkpoints/`, `embeddings/`, `assignments/`, `topics/`, `metrics/`, `figures/` |
| `resultats/malt/` | `best_model.pt`, `exports/`, `evaluation/`, `bn_staging/` |
| `resultats/raw_embedding/`, `batch_triplet/`, `softtriple/`, `supcon/` | Méthodes comparées |
| `resultats/comparisons/` | Tableaux et figures agrégés (`collect_results.py`) |
| `resultats/comparisons/topics_legacy/` | Comparaison topics + `stopwords_domain.txt` |

Migration depuis d'anciens dossiers `runs/` ou `outputs/` :

```bash
python scripts/migrate_legacy_outputs.py --dry-run
python scripts/migrate_legacy_outputs.py
```

## Workflow principal

```bash
cd text

# 1. Embedding brut (référence)
python scripts/export_raw_embeddings.py

# 2. SCGM BTP
sbatch jobs/train_scgm_text.sh
# Post-traitement (emb. test → embeddings/test/, métriques raw BTP+test, export SCGM, projections test, metrics SCGM test, OpenAI optionnel) :
sbatch jobs/postprocess_scgm_text.sh
# Thèmes OpenAI sur login si le nœud GPU n'a pas Internet :
SKIP_OPENAI=0 bash jobs/postprocess_scgm_text.sh
# Sorties clés : resultats/raw_embedding/metrics/metrics_geometry.csv,
#   resultats/raw_embedding_test/metrics/metrics_geometry.csv,
#   resultats/scgm_text/metrics/metrics_geometry_test.csv,
#   embeddings/test/Qwen3-Embedding-0.6B_metallurgie.csv

# 3. MALT (BTP → métallurgie)
python scripts/train_malt_target.py --config configs/malt_btp_to_mettalurgie_qwen06.yaml
python scripts/export_malt_outputs.py \
  --checkpoint resultats/malt/best_model.pt \
  --source_checkpoint resultats/scgm_text/checkpoints/best_model.pt \
  --output_dir resultats/malt/exports

# 4. Méthodes contrastives (entraînement natif, YAML configs/methods/*.yaml)
python scripts/train_batch_triplet.py   # → resultats/<method>/embeddings/final_embeddings.csv
python scripts/train_softtriple.py
python scripts/train_supcon.py
# Recalcul métriques uniquement si besoin :
python scripts/postprocess_contrastive_results.py --method batch_triplet

# 5. Embeddings test (Qwen figé) — inclus dans postprocess ; manuel :
python scripts/export_test_embeddings.py  # → embeddings/test/Qwen3-Embedding-0.6B_metallurgie.csv

# 6. Tuning K-fold (sélection sur mean δ_macro %)
sbatch jobs/tune_scgm_text.sh
sbatch jobs/tune_batch_triplet.sh
# Sorties : resultats/<method>/tuning/grid_summary.csv, best_combo.json
# Puis fit final 100 % BTP → metrics_geometry_btp.csv + metrics_geometry_test.csv

# 7. Agrégation
python scripts/collect_results.py
python scripts/compare_methods.py
```

### Évaluation BTP vs test

| Corpus | Chemin | Encodeur |
|--------|--------|----------|
| BTP (entraînement) | `dataset/data_btp.csv` | Best model fine-tuné (contrastifs) ou tête SCGM + embeddings Qwen figés |
| Test métallurgie | `dataset/test/data_metallurgie.csv` | SCGM : `embeddings/test/Qwen3-Embedding-0.6B_metallurgie.csv` ; métriques raw : `resultats/raw_embedding_test/` |

**Train simple** (`jobs/train_*.sh`, `n_folds: 5`) : (1) K-fold → `kfold_summary.csv` (validation in-domain, μ ± σ) sous `folds/fold_{k}/` ; (2) **fit final 100 % BTP** → `checkpoints/best_model` ; (3) évaluation **BTP + test** → `metrics_geometry_btp.csv`, `metrics_geometry_test.csv` (un seul modèle, pas d’éval test par fold).

**Tuning** (`jobs/tune_*.sh`) : même K-fold par combo pour sélectionner les hyperparamètres, puis **fit final** 100 % BTP + `metrics_geometry_btp.csv` / `metrics_geometry_test.csv`.

### Jobs SLURM

```bash
cd jobs
sbatch train_scgm_text.sh
sbatch postprocess_scgm_text.sh    # après train : emb. test, raw metrics BTP/test, topics, projections, SCGM test metrics (SKIP_OPENAI=1 par défaut)
sbatch train_batch_triplet.sh
# … ou : bash submit_all.sh
sbatch compare_methods.sh
```

**Chaînage train → postprocess** (optionnel) :

```bash
TRAIN_ID=$(sbatch --parsable jobs/train_scgm_text.sh)
sbatch --dependency=afterok:"${TRAIN_ID}" jobs/postprocess_scgm_text.sh
```

Logs SLURM : `jobs/slurm-<job_name>-<job_id>.out` (et `.err`) après `sbatch` depuis `jobs/`. Cache HF : `$SCRATCH/hf_cache` si défini. Jobs GPU : `--constraint='a100|h100'`, `--mem=64G`. Les scripts `jobs/*.sh` utilisent des fins de ligne LF (voir `.gitattributes`).

## SCGM-Text

Macros observées `A0`–`C` ; latents `z` = thèmes intra-macro. Données : `dataset/data_btp.csv` + `embeddings/Qwen3-Embedding-0.6B_btp.csv` (alignement `doc_id`).

**`input_mode`** : `text` (backbone HF fine-tunable) ou `precomputed_embeddings` (colonnes `dim_*`). **`projection`** : `identity` | `linear` | `mlp`. Avec `text` + `freeze_backbone=false`, `identity` signifie **h = f_θ(x)** (pas des embeddings figés).

**Sélection du meilleur checkpoint** (`best_model.pt`) : `val_eta2_macro_balanced` par défaut (pas F1). Option `--best_checkpoint_metric composite` avec `--best_checkpoint_lambda` (score = eta² − λ·C1). Diagnostics classifieur / subtype : `--compute_classifier_diagnostics` / `--compute_subtype_diagnostics` (désactivés par défaut).

Presets d'entraînement :

```bash
python scripts/train_scgm_text.py --config configs/methods/scgm_text.yaml
python scripts/train_scgm_text.py --config configs/scgm_text_strict_finetune_identity.yaml --strict_finetune_identity
```

Thèmes OpenAI (`themes_by_z_openai.csv`, hors notebook — accès Internet requis) :

```bash
# Sur le nœud de login HPC2 (recommandé) :
cd text && bash jobs/enrich_scgm_themes_openai.sh

# Ou en CLI :
python scripts/enrich_scgm_themes_openai.py --output_dir resultats/scgm_text

# Test connectivité seulement :
PROBE_ONLY=1 bash jobs/enrich_scgm_themes_openai.sh
```

Prérequis : `OPENAI_API_KEY` dans `text/.env` ; export SCGM déjà fait (`topics/themes_by_z.csv`).

## MALT-EM

Transfert vers une cible sans label macro dur : responsabilités souples `p0(y|x)` depuis le SCGM source, puis E-step Sinkhorn + M-step EM sur la cible.

```bash
python scripts/evaluate_malt_transfer.py \
  --exports_dir resultats/malt/exports \
  --output_dir resultats/malt/evaluation \
  --label_col pred_label
```

## Comparaison de topics

Notebook `notebooks/03_compare_malt_bertopic_kmeans_topics.ipynb` : MALT + c-TF-IDF vs BERTopic intra-macro vs KMeans intra-macro. Stopwords métier : `resultats/comparisons/topics_legacy/stopwords_domain.txt`.

```bash
python -m nltk.downloader stopwords
jupyter notebook notebooks/03_compare_malt_bertopic_kmeans_topics.ipynb
```

## Réseaux bayésiens

- `notebooks/04_malt_to_bayesian_network.ipynb` — entrée `resultats/malt/exports`, sortie `resultats/malt/bn_staging/`
- `notebooks/04_bayesian_network_from_scgm.ipynb` — entrée `resultats/scgm_text/`, sortie `resultats/scgm_text/bn_staging/`

Dépendances : `numpy<2`, `pgmpy>=0.1.23,<1.0`. Utiliser le même interpréteur Python que le noyau Jupyter (`import sys; print(sys.executable)`).

## Notebooks

Le **corpus** (BTP, métallurgie, etc.) est défini dans les cellules *Parameters* ou les YAML `configs/methods/`, pas dans le nom du fichier `.ipynb`.

| Notebook | Rôle |
|----------|------|
| `00_check_data.ipynb` | Aperçu du CSV configuré |
| `01_compare_embedding_methods.ipynb` | Comparaison globale eta² / RankMe |
| `01_scgm_text_experiment.ipynb` | **Lecture seule** — §3 chargement ; §4 K-fold (tables) ; §5 BTP ; §6 test (SCGM + raw). Après `postprocess_scgm_text.sh` (métriques raw BTP/test incluses). |
| `02_malt_btp_to_mettalurgie_transfer.ipynb` | **Lecture seule** — analyse MALT à partir de `resultats/malt/` |
| `03_compare_malt_bertopic_kmeans_topics.ipynb` | Qualité topics |
| `04_malt_to_bayesian_network.ipynb` | BN depuis MALT |
| `04_bayesian_network_from_scgm.ipynb` | BN depuis SCGM |
| `05_view_batch_triplet_results.ipynb` | Résultats Batch Triplet (`resultats/batch_triplet/`) — métriques + **PCA/t-SNE** BTP et test (macro + centroïdes) si `embeddings/final_embeddings_*.csv` présents |
| `05_view_softtriple_results.ipynb` | Résultats SoftTriple (idem) |
| `05_view_supcon_results.ipynb` | Résultats SupCon (idem) |

`01_draft.ipynb` : brouillon obsolète — ne pas utiliser.

Entraînement **hors notebook** : `scripts/train_scgm_text.py`, `scripts/train_malt_target.py`, ou `jobs/*.sh` (SLURM). Les notebooks chargent checkpoints, `train_log.csv` et exports déjà produits.

**JupyterHub (HPC2)** : JupyterLab ne voit que `~/notebooks`. Créer un lien vers le projet, par ex. `ln -sfn ~/SAFER/text ~/notebooks/SAFER_text`, puis kernel Python avec le venv du projet (`ipykernel install --user --name safer-text`).

Les fichiers `notebooks/*.ipynb` ne sont **pas versionnés** (restent sur la machine / le cluster). Après `git pull`, régénérer :

```bash
python scripts/build_analysis_notebooks.py   # 00, 01_compare, 05_view_*
python scripts/rebuild_notebook_01.py   # notebook 01 : §3–§6 (viz allégées ; §5g → export_raw_embeddings.py)
python scripts/build_malt_notebook.py
python scripts/build_notebook_03_compare_topics.py
python scripts/build_notebook_04_malt_bn.py
python scripts/build_notebook_04_bn_btp_from_scgm.py
```

## Métriques principales

- **eta2_macro_balanced**, **eta2_weighted** (`metrics/inertia.py`) — structuration des macros sur distance euclidienne
- **rankme_global**, **c1_global**, **c10_global** — géométrie des embeddings

Pas d'Accuracy / F1 / NMI dans le tableau principal de comparaison des méthodes.

## Prompts

Pipeline principal : `text_col=sentence`, `use_prompt: false` dans toutes les configs `configs/methods/`.

## Méthodes contrastives

**Métrique principale** : δ_macro (%) = `delta_macro_pct` = 100 × η²_macro_balanced (structuration macro de l'espace). Compléments : `rankme_global`, `c1_global`, `c10_global`. Sélection du meilleur checkpoint sur le **val** via δ_macro (plus `eval_loss`).

**Distance d'entraînement** (défaut `training.distance_metric: euclidean`) : SupCon (−‖z_i−z_j‖²/τ), SoftTriple (−‖z−c‖² vers centroïdes), batch triplet (`BatchHardSoftMarginTripletLoss` euclidienne). Les métriques val/export restent η² sur distance euclidienne² (embeddings L2-normalisés à l'encode).

### Expérience single-run (configs inchangées)

`configs/methods/batch_triplet.yaml`, `softtriple.yaml`, `supcon.yaml` — jobs `train_*.sh` :

```bash
python scripts/train_batch_triplet.py --config configs/methods/batch_triplet.yaml
# K=5 par défaut (n_folds dans le YAML) → resultats/batch_triplet/metrics/kfold_summary.csv
```

Pour un split unique (ancien comportement) : `n_folds: 1` dans le YAML.

### Tuning (grille + réentraînement final 100 %)

YAML dédiés sous `configs/tuning/` (ne modifient pas les configs `methods/`) :

```bash
python scripts/tune_batch_triplet.py --grid-config configs/tuning/batch_triplet_grid.yaml
# ou : sbatch jobs/tune_batch_triplet.sh
```

Sorties tuning : `resultats/<method>/tuning/grid_summary.csv`, `best_combo.json`, `combos/<combo_id>/`.  
Après tuning : réentraînement sur tout le corpus → `resultats/<method>/embeddings/final_embeddings.csv`.

Package : `contrastive_methods/` (`train.py`, `tuning.py`, `training_*.py`, `eval_geometry.py`).

## Tests

```bash
python -m pytest tests/
```
