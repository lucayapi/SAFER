# Pipeline SAFER — texte

Analyse de récits d'accidents : SCGM-G sur embeddings BTP, méthodes contrastives, comparaison d'embeddings et réseaux bayésiens exploratoires (exports SCGM).

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
| `dataset/test/` | Corpus test hors domaine (registre `configs/test_corpora.yaml`) |
| `embeddings/` | Embeddings pré-calculés (local, gitignored) |
| `configs/` | `paths.yaml`, `methods/*.yaml`, configs SCGM / contrastifs |
| `safer_core/` | Chemins centralisés → `output/` |
| `scgm_text/` | Modèle et entraînement SCGM-G texte |
| `contrastive_methods/` | Batch Triplet, SoftTriple, SupCon |
| `learn_embeddings/` | Export embeddings bruts (encodeur figé) |
| `bn_pipeline/` | Réseaux bayésiens (pgmpy, exports SCGM) |
| `macro_transfer/` | Transfert macro-guidé + topics intra-macro (cible test) |
| `scripts/` | CLI entraînement, export, évaluation, agrégation |
| `jobs/` | Scripts SLURM Mésocentre |
| `notebooks/` | Analyse (**.ipynb gitignored**, régénération locale via `scripts/build_*.py`) |
| `legacy/` | Code historique (anciens jobs, hors contrastif) |
| `output/` | **Toutes les sorties** (gitignored) |

## Sorties (`output/`)

| Chemin | Contenu |
|--------|---------|
| `output/scgm_text/` | `checkpoints/`, `embeddings/`, `assignments/`, `topics/`, `metrics/`, `figures/`, `bn_staging/` |
| `output/raw_embedding/`, `batch_triplet/`, `softtriple/`, `supcon/` | Méthodes comparées |
| `output/comparisons/` | Tableaux et figures agrégés (`collect_results.py`) |

Migration depuis d'anciens dossiers `runs/` ou `outputs/` :

```bash
python scripts/migrate_legacy_outputs.py --dry-run
python scripts/migrate_legacy_outputs.py
```

## Workflow principal

```bash
cd text

# 1. Embedding brut BTP + test (géométrie)
sbatch jobs/export_raw_geometry.sh
# ou : python scripts/export_raw_embeddings.py

# 2. SCGM BTP (+ eval test → output_test/ en fin de train)
sbatch jobs/train_scgm_text.sh
sbatch jobs/export_test_embeddings.sh   # CSV Qwen test si absent
# Sorties clés : output/raw_embedding/metrics/metrics_geometry.csv,
#   output_test/<corpus>/raw_embedding/metrics/metrics_geometry.csv,
#   output_test/<corpus>/scgm_text/metrics/metrics_geometry_test.csv,
#   embeddings/test/Qwen3-Embedding-0.6B_<corpus>.csv

# 3. Méthodes contrastives (entraînement natif, YAML configs/methods/*.yaml)
python scripts/train_batch_triplet.py   # → output/<method>/embeddings/final_embeddings.csv
python scripts/train_softtriple.py
python scripts/train_supcon.py
# Recalcul métriques uniquement si besoin :
python scripts/postprocess_contrastive_results.py --method batch_triplet

# 4. Embeddings test (Qwen figé) — job ou CLI :
sbatch jobs/export_test_embeddings.sh
# python scripts/export_test_embeddings.py --corpus metallurgie

# 5. Tuning K-fold (sélection sur mean eta2_macro_balanced_perc)
cd jobs && bash submit_tuning_all.sh
# ou : sbatch tune_scgm_text.sh tune_softtriple.sh tune_supcon.sh tune_batch_triplet.sh
# Variables : TEST_CORPUS, MAX_COMBOS, SKIP_FINAL_FIT, SEED, GRID_CONFIG
# Sorties : output/<method>/tuning/grid_summary.csv, best_combo.json
# Puis fit final 100 % BTP → metrics_geometry_btp.csv + metrics_geometry_test.csv

# 6. Agrégation (tableaux BTP + test → notebooks/01_compare_embedding_methods.ipynb)
python scripts/collect_results.py
# → comparisons/tables/embedding_geometry_comparison_btp.csv (fit final)
# → comparisons/tables/embedding_geometry_comparison_btp_kfold.csv (μ±σ val K-fold)
python scripts/compare_methods.py
python scripts/compare_methods.py --corpus test
```

### Évaluation BTP vs test

| Corpus | Chemin | Encodeur |
|--------|--------|----------|
| BTP (entraînement) | `dataset/data_btp.csv` | Best model fine-tuné (contrastifs) ou tête SCGM + embeddings Qwen figés |
| Test métallurgie | `dataset/test/data_metallurgie.csv` | SCGM : `embeddings/test/Qwen3-Embedding-0.6B_metallurgie.csv` ; métriques raw : `output_test/metallurgie/raw_embedding/` |

**Train simple** (`jobs/train_*.sh`, `n_folds: 5`) : (1) K-fold → `kfold_summary.csv` (validation in-domain, **mean/std** sur toutes les métriques géométriques et sur `train_wall_time_sec` par fold) sous `folds/fold_{k}/` ; (2) **fit final 100 % BTP** → `checkpoints/best_model` + colonne `final_fit_wall_time_sec` dans `kfold_summary.csv` ; (3) évaluation **BTP + test** → `metrics_geometry_btp.csv`, `metrics_geometry_test.csv` (un seul modèle, pas d’éval test par fold).

**Tuning** (`jobs/tune_*.sh`) : même K-fold par combo → `grid_summary.csv` reprend les colonnes `mean_*` / `std_*` (géométrie + temps fold) ; le fit final n’est pas rejoué par combo (sélection puis fit final du meilleur combo).

### Jobs SLURM

```bash
cd jobs
sbatch export_raw_geometry.sh       # métriques embedding brut BTP + test (parallèle au train OK)
sbatch train_scgm_text.sh         # inclut eval BTP + test
sbatch export_test_embeddings.sh  # CSV Qwen test si besoin
sbatch train_batch_triplet.sh
# … ou : bash submit_all.sh
CORPUS=metallurgie bash run_tpn_macro_transfer.sh
sbatch compare_methods.sh
```

Logs SLURM : `jobs/slurm-<job_name>-<job_id>.out` (et `.err`) après `sbatch` depuis `jobs/`. Cache HF : `$SCRATCH/hf_cache` si défini. Jobs GPU : `--constraint='a100|h100'`, `--mem=64G`. Les scripts `jobs/*.sh` utilisent des fins de ligne LF (voir `.gitattributes`).

## SCGM-Text

Macros observées `A0`–`C` ; latents `z` = thèmes intra-macro. Données : `dataset/data_btp.csv` + `embeddings/Qwen3-Embedding-0.6B_btp.csv` (alignement `doc_id`).

Pipeline **end2end** : texte → Qwen (`backbone_trainable` / `train_last_n_layers` dans la config) → projecteur (`linear` | `mlp`) → tête SCGM.

**Config unique** : [`configs/scgm_text.yaml`](configs/scgm_text.yaml) — modes backbone documentés en tête du fichier (gelé / k dernières couches / complet).

**Sélection du meilleur checkpoint** (`best_model.pt`) : `val_eta2_macro_balanced` par défaut (pas F1). Option `--best_checkpoint_metric composite` avec `--best_checkpoint_lambda` (score = eta² − λ·C1). Diagnostics classifieur / subtype : `--compute_classifier_diagnostics` / `--compute_subtype_diagnostics` (désactivés par défaut).

```bash
python scripts/train_scgm_text.py --config configs/scgm_text.yaml
# ou
sbatch jobs/train_scgm_text.sh
```

**Topics** : uniquement via **macro_transfer** sur corpus test (notebook 06). Export SCGM BTP complet (`export_scgm_text_outputs.py`) reste en CLI manuelle si besoin.

## Corpus de test (configurable)

Registre : [`configs/test_corpora.yaml`](configs/test_corpora.yaml) — chaque entrée définit `data_csv`, `emb_csv`, `display_name`. Défaut : **`metallurgie`**.

**Ajouter un corpus** : fichiers `dataset/test/data_<id>.csv` + `embeddings/test/Qwen3-Embedding-0.6B_<id>.csv`, entrée dans le registre, puis :

```bash
export TEST_CORPUS=<id>   # ou CORPUS= pour macro_transfer
python scripts/export_test_embeddings.py --corpus <id>
```

Utilisé par : entraînement SCGM, contrastifs, jobs raw/test emb, notebooks 01 / 05 / 06 / 07 (`TEST_CORPUS`).

### Arborescence `output` vs `output_test`

| Racine | Contenu |
|--------|---------|
| `output/<method>/` | **BTP uniquement** : checkpoints, métriques BTP, exports BTP (sans topics test) |
| `output_test/<corpus>/<method>/` | Métriques / projections **test** (`metrics_geometry_test.csv`, …) |
| `output_test/<corpus>/raw_embedding/` | Embedding brut test |
| `output_test/<corpus>/macro_transfer/<method>/` | Transfert macro + topics BERTopic (`theme_label` via `bertopic.representation` par défaut) |
| `output_test/<corpus>/bn_staging/` | Sorties notebook 04 (BN) |

Les anciens dossiers `resultats/` et `resultats_test/` peuvent être renommés en `output/` et `output_test/` (ou relancer les jobs). Les chemins legacy `resultats/...` sont redirigés vers `output/...` si `SAFER_LEGACY_PATHS=1`.

## Transfert macro TPN + topics cible (corpus test)

Pipeline **`macro_transfer/`** (TPN) : encodeur contrastif BTP **gelé** → adaptateur prototypique sur la cible → gating macro \(p(m|u)\) → topics **BERTopic** intra-macro (UMAP, HDBSCAN, c-TF-IDF, `stop_metier.txt`), libellés via **`bertopic.representation.OpenAI`** par défaut.

**Checkpoints source** (BTP) : `output/<encodeur>/checkpoints/` (ex. `softtriple`, `scgm_text`, `supcon`).

Paramètres : [`configs/tpn_macro_transfer.yaml`](configs/tpn_macro_transfer.yaml). Encodeur choisi via `BASE_METHOD` dans `jobs/run_tpn_macro_transfer.sh`.

**Libellés topics** : macros A0–C ([`configs/accident_macros.yaml`](configs/accident_macros.yaml)) + contexte sectoriel ([`configs/corpus_prompt_context.yaml`](configs/corpus_prompt_context.yaml)).

| Étape | Commande |
|-------|----------|
| Transfert TPN | `CORPUS=<id> bash jobs/run_tpn_macro_transfer.sh` (défaut `scgm_text`) |
| Encodeur | `BASE_METHOD=softtriple CORPUS=<id> bash jobs/run_tpn_macro_transfer.sh` |
| CLI | `python scripts/run_tpn_macro_transfer_discovery.py --corpus <id>` |
| BERTopic seul | `--bertopic-only` (artefacts déjà encodés) |
| Notebook 06 | comparaison `tpn_scgm_text` vs `tpn_softtriple` |
| Notebook 07 | run local TPN + viz 2D |
| Notebook 08 | diagnostics TPN (initial vs adapté) |

**Sorties** : `output_test/<corpus_id>/macro_transfer/tpn_<encodeur>/`

## Réseaux bayésiens

- `notebooks/04_bayesian_network_macro_transfer.ipynb` — staging depuis **macro_transfer** (corpus test), sortie `output_test/<TEST_CORPUS>/bn_staging/`

Dépendances : `numpy<2`, `pgmpy>=0.1.23,<1.0`. Utiliser le même interpréteur Python que le noyau Jupyter (`import sys; print(sys.executable)`).

## Notebooks

Le **corpus** (BTP, métallurgie, etc.) est défini dans les cellules *Parameters* ou les YAML `configs/methods/`, pas dans le nom du fichier `.ipynb`.

| Notebook | Rôle |
|----------|------|
| `00_check_data.ipynb` | Aperçu du CSV configuré |
| `01_compare_embedding_methods.ipynb` | Comparaison **Embedding brut + Batch Triplet / SupCon / SoftTriple / SCGM** — BTP **K-fold (μ±σ)**, BTP fit final, test métallurgie ; η² / RankMe |
| `02_scgm_text_results.ipynb` | **Lecture seule** — BTP (`output/scgm_text`) ; test (`output_test/<TEST_CORPUS>/scgm_text`). Topics : notebook 06 / `run_tpn_macro_transfer.sh`. |
| `04_bayesian_network_macro_transfer.ipynb` | BN sur corpus test (staging **macro_transfer** → `output_test/<TEST_CORPUS>/bn_staging/`) |
| `05_view_batch_triplet_results.ipynb` | Résultats Batch Triplet (`output/batch_triplet/`) — métriques + **PCA/t-SNE** BTP et test (macro + centroïdes) si `embeddings/final_embeddings_*.csv` présents |
| `05_view_softtriple_results.ipynb` | Résultats SoftTriple (idem) |
| `05_view_supcon_results.ipynb` | Résultats SupCon (idem) |
| `06_macro_transfer_topics.ipynb` | **Lecture seule** — comparaison TPN SCGM vs SoftTriple + topics BERTopic |
| `07_macro_transfer_interactive.ipynb` | **Run local** TPN (`BASE_METHOD`) + viz 2D ; `RUN_PIPELINE=False` pour reviz sans refit |
| `08_tpn_macro_transfer_results.ipynb` | **Lecture seule** — métriques initial/adapté, gating, PCA, courbe d’entraînement |

`01_draft.ipynb` : brouillon obsolète — ne pas utiliser.

Entraînement **hors notebook** : `scripts/train_scgm_text.py` ou `jobs/*.sh` (SLURM). Les notebooks chargent checkpoints, `train_log.csv` et exports déjà produits.

**JupyterHub (HPC2)** : JupyterLab ne voit que `~/notebooks`. Créer un lien vers le projet, par ex. `ln -sfn ~/SAFER/text ~/notebooks/SAFER_text`, puis kernel Python avec le venv du projet (`ipykernel install --user --name safer-text`).

Les fichiers `notebooks/*.ipynb` ne sont **pas versionnés** (restent sur la machine / le cluster). Après `git pull`, régénérer :

```bash
python scripts/build_analysis_notebooks.py        # 00, 01_compare, 05_view_*
python scripts/build_notebook_02_scgm_results.py  # 02_scgm_text_results
python scripts/build_notebook_04_bn_macro_transfer.py
python scripts/build_notebook_06_macro_transfer_topics.py
python scripts/build_notebook_07_macro_transfer_interactive.py
```

## Métriques principales

- **eta2_macro_balanced**, **eta2_weighted** (`metrics/inertia.py`) — structuration des macros sur distance euclidienne
- **rankme_global**, **c1_global**, **c10_global** — géométrie des embeddings

Pas d'Accuracy / F1 / NMI dans le tableau principal de comparaison des méthodes.

## Prompts

Pipeline principal : `text_col=sentence`, `use_prompt: false` dans toutes les configs `configs/methods/`.

## Méthodes contrastives

**Métrique principale** : δ_macro (%) = `eta2_macro_balanced_perc` = 100 × η²_macro_balanced (structuration macro de l'espace). Compléments : `rankme_global`, `c1_global`, `c10_global`. Sélection du meilleur checkpoint sur le **val** via δ_macro (plus `eval_loss`).

**Losses d'entraînement** :
- **Batch triplet** : [`BatchHardSoftMarginTripletLoss`](https://sbert.net/docs/sentence_transformer/loss_overview.html) (Sentence Transformers) + sampler `GROUP_BY_LABEL` ; `training.distance_metric` = euclidien par défaut.
- **SupCon** : [HobbitLong/SupContrast](https://github.com/HobbitLong/SupContrast) (`SupConLoss`, cosinus L2, 1 vue par phrase) ; `training.distance_metric` doit être `cosine` ; hyperparamètres dans `supcon:` (`temperature`, `base_temperature`, `contrast_mode`).
- **SoftTriple** : loss native ; euclidien par défaut. Entraînement custom avec **AMP GPU** (bf16/fp16, aligné ST), val `loss` + géométrie en **une passe** par epoch. Console : `[SoftTriple epoch=k/N] train_loss=… | val_loss=… | …` ; détail dans `metrics/train_log.csv`.

Les métriques val/export (η², RankMe) restent calculées sur embeddings L2-normalisés, indépendamment de la loss.

### SoftTriple — régularisation des centres et centres effectifs

`centers_per_class` (**K**) est le **nombre maximal initial** de centres latents par macro-classe, pas le nombre final de sous-classes.

| `center_regularization_type` | Effet |
|------------------------------|--------|
| `none` | K centres fixes, pas de régularisation (`tau: 0`) |
| `diversity` | Régularisation historique : éloigne les centres trop proches (marges `center_min_distance` / `center_max_similarity`) |
| `merge_l21` | Régularisation inspirée de l’article SoftTriple : minimise les distances intra-classe entre centres pour encourager la fusion des centres redondants |

**Rétrocompatibilité** : si `center_regularization_type` est absent du YAML, `tau <= 0` → `none`, sinon → `diversity` (comportement des runs existants avec `softtriple.yaml`).

Dans la variante `merge_l21`, K ne doit pas être interprété comme le nombre final de sous-classes : après apprentissage, les centres proches sont regroupés (composantes connexes) pour estimer un **nombre effectif de centres uniques** par macro.

Configs d’expérience :

```bash
python scripts/train_softtriple.py --config configs/methods/softtriple_no_reg.yaml
python scripts/train_softtriple.py --config configs/methods/softtriple_merge_l21.yaml
python scripts/train_softtriple.py --config configs/methods/softtriple_diversity.yaml
```

Export (si `export_effective_centers: true` et régularisation ≠ `none`) dans `{output_dir}/centers/` et `checkpoints/best_model/centers/` :

- `softtriple_centers_raw.pt`, `softtriple_effective_centers.pt`
- `softtriple_effective_centers.csv`, `softtriple_center_assignments.csv`
- `softtriple_center_diagnostics.json` (effectifs par macro, hyperparamètres)

Grille tuning : `softtriple.center_regularization_type` × `softtriple.tau` — combinaisons invalides filtrées (`none` → `tau=0`, `merge_l21` exclut `tau<=0`).

### Expérience single-run (configs inchangées)

`configs/methods/batch_triplet.yaml`, `softtriple.yaml`, `supcon.yaml` — jobs `train_*.sh` :

```bash
python scripts/train_batch_triplet.py --config configs/methods/batch_triplet.yaml
# K=5 par défaut (n_folds dans le YAML) → output/batch_triplet/metrics/kfold_summary.csv
```

Pour un split unique (ancien comportement) : `n_folds: 1` dans le YAML.

**PKBatchSampler (Batch Triplet)** : batches équilibrés auto — P = nombre de macros dans le fold train, K = `batch_size / P` (ex. 64 et 4 classes → 16 ex./classe). Vérification : `python scripts/debug_pk_sampler.py`. Au démarrage : `[PKSampler DEBUG] batch=0 labels={"0":16,...}`.

**SupCon** : sampler shuffle standard ST (`BATCH_SAMPLER`), pas PK.

**Batch Triplet / SupCon** : loss native ST. Pendant l’entraînement : `[BatchTriplet epoch=k/N] train_loss=… | eta2_macro_balanced_perc=…` (idem `SupCon`) ; détail dans `metrics/train_log.csv`. Un dict HF récapitulatif peut encore apparaître à la fin de `trainer.train()` (pas un log par epoch).

### Tuning (grille + réentraînement final 100 %)

YAML dédiés sous `configs/tuning/` (ne modifient pas les configs `methods/`) :

```bash
python scripts/tune_batch_triplet.py --grid-config configs/tuning/batch_triplet_grid.yaml
# ou : sbatch jobs/tune_batch_triplet.sh
# Limiter la grille : MAX_COMBOS=8 sbatch jobs/tune_softtriple.sh
# K-fold seul : SKIP_FINAL_FIT=1 sbatch jobs/tune_supcon.sh
```

Grille en **notation pointée** (`training.learning_rate`, `supcon.temperature`, `softtriple.gamma`, `training.distance_metric`, etc.). Hyperparamètres spécifiques sous `supcon:` / `softtriple:` / `batch_triplet:` uniquement (pas de `distance_metric` dupliqué).

**Log d'entraînement** : `metrics/train_log.csv` — `epoch`, `train_loss`, `val_loss` (SoftTriple), puis `val_{k}` pour chaque `k` dans `GEOMETRY_METRIC_KEYS` (aligné sur `metrics_geometry_*.csv`). Jobs SLURM : cache HF via `HF_HOME` uniquement (`jobs/_env.sh`, pas `TRANSFORMERS_CACHE`).

Sorties tuning : `output/<method>/tuning/grid_summary.csv`, `best_combo.json`, `combos/<combo_id>/`.  
Après tuning : réentraînement sur tout le corpus → `output/<method>/embeddings/final_embeddings.csv`.

Package : `contrastive_methods/` (`train.py`, `tuning.py`, `training_*.py`, `eval_geometry.py`, `training_log.py`).

## Tests

```bash
python -m pytest tests/
```
