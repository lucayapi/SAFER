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

Variables d'environnement : `HF_TOKEN` ou `HUGGING_FACE_HUB_TOKEN` dans `.env` (modèles Hugging Face). `OPENAI_API_KEY` pour l'enrichissement de thèmes et l'annotation (`annotation/`). Ne jamais committer `.env`.

## Organisation

| Dossier | Rôle |
|---------|------|
| `dataset/` | CSV métadonnées annotées (`data_btp.csv`, `data_metallurgie.csv`, `data_caou.csv`) |
| `embeddings/` | Embeddings pré-calculés (local, gitignored) |
| `configs/` | `paths.yaml`, `methods/*.yaml`, configs SCGM / contrastifs |
| `safer_core/` | Chemins centralisés → `output/` |
| `scgm_text/` | Modèle et entraînement SCGM-G texte |
| `contrastive_methods/` | Batch Triplet, SoftTriple, SupCon |
| `bn_pipeline/` | Réseaux bayésiens (pgmpy, exports SCGM) |
| `macro_transfer/` | Transfert macro-guidé + topics intra-macro (cible test) |
| `annotation/` | Annotation unités factuelles via API OpenAI (notebook + cache JSONL + prompt caching) |
| `scripts/` | CLI entraînement, export, évaluation, agrégation |
| `jobs/` | Scripts SLURM Mésocentre |
| `notebooks/` | Analyse (**.ipynb gitignored**, régénération locale via `scripts/build_*.py`) |
| `output/` | **Toutes les sorties** (gitignored) |

## Annotation (API OpenAI)

Notebook interactif pour annoter des unités factuelles (A0/A1/B/C + injury/hospitalized/fatal) avec reprise sur cache JSONL et **prompt caching OpenAI** pour réduire le coût du préfixe système répété.

| Étape | Commande / fichier |
|-------|-------------------|
| Données d'entrée | Placer un CSV dans `annotation/data/` (colonnes : `accident_id`, `sentence`, `fact_id`, `accident_summary`) |
| Notebook | `annotation/annotate_factual_units.ipynb` (régénérer : `python scripts/build_notebook_annotation.py`) |
| Sorties | `annotation/outputs/<run_id>/` — JSONL cache, XLSX annoté, summary, accident_outcomes |
| Clé API | `OPENAI_API_KEY` dans `text/.env` |

**Deux types de cache (distincts) :**

| Mécanisme | Rôle | Contrôle |
|-----------|------|----------|
| **Cache JSONL local** | Reprendre une run sans rappeler l'API pour les lignes déjà annotées | `SKIP_CACHE`, `RUN_ID`, fichier `*.jsonl` dans `outputs/` |
| **Prompt caching OpenAI** | Réutiliser côté serveur le préfixe identique (`SYSTEM_PROMPT`) via `prompt_cache_key` | `USE_PROMPT_CACHE_KEY`, `PROMPT_CACHE_KEY` (défaut : `safer-annotation:{PROMPT_VERSION}`) |

Le runner loggue `cached_tokens` (depuis `usage.prompt_tokens_details`) et le taux `cached_tokens / prompt_tokens` en fin de run. Modèle par défaut : `gpt-5.4-mini` (`reasoning_effort=medium`, `max_output_tokens=4000`).

Paramètres principaux (cellule dédiée du notebook) : `INPUT_CSV`, `OPENAI_MODEL`, `PROMPT_CACHE_KEY`, `USE_PROMPT_CACHE_KEY`, `N_ACCIDENTS`, `UNITS_PER_ACCIDENT`, `SKIP_CACHE`, `DRY_RUN`, `MIN_DELAY_BETWEEN_CALLS_SEC`, `RUN_ID` (reprise d'une run existante).

### Mode Batch OpenAI (asynchrone, moins cher)

Pour de gros volumes, utiliser l'**API Batch** (`/v1/chat/completions`) avec config YAML et CLI :

| Étape | Commande / fichier |
|-------|-------------------|
| Config | `configs/annotation_batch.yaml` (mêmes paramètres que le notebook synchrone) |
| Soumission | `python scripts/run_annotation_batch.py submit --config configs/annotation_batch.yaml` |
| Suivi statut | `python scripts/run_annotation_batch.py status --run-id <RUN_ID>` ou notebook `annotation/check_batch_status.ipynb` |
| Téléchargement | `python scripts/run_annotation_batch.py download --run-id <RUN_ID>` (quand `status=completed`) |
| Ingestion XLSX | `python scripts/run_annotation_batch.py ingest --run-id <RUN_ID>` |
| Pipeline complet | `python scripts/run_annotation_batch.py pipeline --config configs/annotation_batch.yaml` |

Fichiers batch dans `annotation/outputs/<run_id>/` :

| Fichier | Rôle |
|---------|------|
| `batch_state.json` | `batch_id`, statut, compteurs OpenAI |
| `{model}__{prompt}__batch_input.jsonl` | Requêtes soumises (`custom_id = accident_id\|\|fact_id`) |
| `batch_output.jsonl` | Réponses OpenAI |
| `batch_errors.jsonl` | Échecs (si présent) |
| `*__annotated.xlsx`, cache `*.jsonl` | Produits finaux après `ingest` (identiques au mode synchrone) |

Régénérer le notebook de suivi : `python scripts/build_notebook_annotation_batch_status.py`

### Annotation deux passes v13

Workflow pour annoter ~43k unités avec détection d'ambiguïté puis désambiguïsation ciblée :

| Passe | Mode | Fichier / commande |
|-------|------|-------------------|
| **1** | Batch (unité seule, sans récit) | `python scripts/run_annotation_batch.py pipeline --config configs/annotation_batch_pass1_v13.yaml` |
| Suivi batch | — | `annotation/check_batch_status.ipynb` avec le `RUN_ID` passe 1 |
| **2** | Notebook synchrone (récit complet pour ambigus) | `annotation/annotate_pass2_ambiguous.ipynb` (régénérer : `python scripts/build_notebook_annotation_pass2.py`) |

**Passe 1** produit notamment : `pred_ambiguous`, `pred_context_needed`, `pred_alternative_label`, `pred_ambiguity_type`, `pred_ambiguity_reason`.

**Passe 2** filtre via `should_run_second_pass()` (unités où `context_needed`, `ambiguous` ou `alternative_label != NONE`), ré-annote avec `accident_summary`, puis fusionne dans `*__annotated_final.xlsx`. Les champs `injury_mentioned` / `hospitalized` / `fatal` restent ceux de la passe 1.

Config passe 1 : `configs/annotation_batch_pass1_v13.yaml` (`prompt_version: v13_two_pass_ambiguity_context`, `pass_mode: pass1`).

Dans le notebook passe 2, renseigner `PASS1_RUN_ID` avec le run batch ingéré.

### Migration vers `dataset/`

Après ingest batch (ou passe 2), exporter les XLSX annotés vers les CSV du pipeline :

```bash
cd text
python scripts/migrate_annotation_to_dataset.py --dry-run
python scripts/migrate_annotation_to_dataset.py --validate-load
# Passe 1 seulement :
python scripts/migrate_annotation_to_dataset.py --pass1-only --run-id run_all_btp
```

Mapping : `run_all_btp` → `dataset/data_btp.csv`, `run_all_metallurgie` → `data_metallurgie.csv`, `run_all_caou_chimie_plas` → `data_caou.csv`. Préfère `*__annotated_final.xlsx` si présent. Puis relancer `export_corpus_embeddings.sh` (`FORCE=1` si embeddings déjà présents).

## Sorties (`output/`)

| Chemin | Contenu |
|--------|---------|
| `output/scgm_text/` | `checkpoints/`, `embeddings/`, `assignments/`, `topics/`, `metrics/`, `figures/`, `bn_results/` |
| `output/raw_embedding/`, `batch_triplet/`, `softtriple/`, `supcon/` | Méthodes comparées |
| `output/comparisons/` | Tableaux et figures agrégés (`collect_results.py`) |

## Workflow principal

```bash
cd text

# 1. Embedding brut BTP + test (géométrie)
sbatch jobs/export_raw_geometry.sh
# ou : python scripts/export_raw_embeddings.py

# 2. SCGM BTP (+ eval test → output_test/ en fin de train)
sbatch jobs/train_scgm_text.sh
sbatch jobs/export_corpus_embeddings.sh   # embeddings Qwen (btp + metallurgie + caou)
# Sorties clés : output/raw_embedding/metrics/metrics_geometry.csv,
#   output_test/<corpus>/raw_embedding/metrics/metrics_geometry.csv,
#   output_test/<corpus>/scgm_text/metrics/metrics_geometry_test.csv,
#   embeddings/Qwen3-Embedding-0.6B_<corpus>.csv

# 3. Méthodes contrastives (entraînement natif, YAML configs/methods/*.yaml)
python scripts/train_batch_triplet.py   # → output/<method>/embeddings/final_embeddings.csv
python scripts/train_softtriple.py
python scripts/train_supcon.py
# Recalcul métriques uniquement si besoin :
python scripts/postprocess_contrastive_results.py --method batch_triplet

# 4. Embeddings encodeur figé (Qwen) — job ou CLI :
sbatch jobs/export_corpus_embeddings.sh
# CORPUS=metallurgie sbatch jobs/export_corpus_embeddings.sh
# python scripts/export_corpus_embeddings.py --config configs/export_embeddings.yaml --corpus metallurgie

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
| BTP (entraînement) | `dataset/data_btp.csv` | Best model fine-tuné (contrastifs) ou tête SCGM + `embeddings/Qwen3-Embedding-0.6B_btp.csv` |
| Métallurgie (test) | `dataset/data_metallurgie.csv` | `embeddings/Qwen3-Embedding-0.6B_metallurgie.csv` ; métriques raw : `output_test/metallurgie/raw_embedding/` |
| Caou (test) | `dataset/data_caou.csv` | `embeddings/Qwen3-Embedding-0.6B_caou.csv` |

**Train simple** (`jobs/train_*.sh`, `n_folds: 5`) : (1) K-fold → `kfold_summary.csv` (validation in-domain, **mean/std** sur toutes les métriques géométriques et sur `train_wall_time_sec` par fold) sous `folds/fold_{k}/` ; (2) **fit final 100 % BTP** → `checkpoints/best_model` + colonne `final_fit_wall_time_sec` dans `kfold_summary.csv` ; (3) évaluation **BTP + test** → `metrics_geometry_btp.csv`, `metrics_geometry_test.csv` (un seul modèle, pas d’éval test par fold).

**Tuning** (`jobs/tune_*.sh`) : même K-fold par combo → `grid_summary.csv` reprend les colonnes `mean_*` / `std_*` (géométrie + temps fold) ; le fit final n’est pas rejoué par combo (sélection puis fit final du meilleur combo).

### Jobs SLURM

```bash
cd jobs
sbatch export_raw_geometry.sh       # métriques embedding brut BTP + test (parallèle au train OK)
sbatch train_scgm_text.sh         # inclut eval BTP + test
sbatch export_corpus_embeddings.sh  # CSV Qwen si absents
sbatch train_batch_triplet.sh
# … ou : bash submit_all.sh
BASE_METHOD=scgm_text CORPUS=metallurgie bash run_frozen_source_prototypes.sh
sbatch compare_methods.sh
```

Logs SLURM : `jobs/slurm-<job_name>-<job_id>.out` (et `.err`) après `sbatch` depuis `jobs/`. Cache HF : `$SCRATCH/hf_cache` si défini. Jobs GPU : `--constraint='a100|h100'`, `--mem=64G`. Les scripts `jobs/*.sh` utilisent des fins de ligne LF (voir `.gitattributes`).

## SCGM-Text

Macros observées `A0`–`C` ; latents `z` = thèmes intra-macro. Données : `dataset/data_btp.csv` + `embeddings/Qwen3-Embedding-0.6B_btp.csv` (alignement `doc_id`).

Pipeline **end2end** : texte → Qwen (`backbone_trainable` / `train_last_n_layers` dans la config) → projecteur (`linear` | `mlp`) → tête SCGM.

**Config unique** : [`configs/methods/scgm_text.yaml`](configs/methods/scgm_text.yaml) — modes backbone documentés en tête du fichier (gelé / k dernières couches / complet).

**Sélection du meilleur checkpoint** (`best_model.pt`) : `val_eta2_macro_balanced` par défaut (pas F1). Diagnostics classifieur / subtype : `--compute_classifier_diagnostics` / `--compute_subtype_diagnostics` (désactivés par défaut).

```bash
python scripts/train_scgm_text.py --config configs/methods/scgm_text.yaml
# ou
sbatch jobs/train_scgm_text.sh
```

**Topics** : uniquement via **macro_transfer** sur corpus test (notebook 06). Export SCGM BTP complet (`export_scgm_text_outputs.py`) reste en CLI manuelle si besoin.

## Corpus de test (configurable)

Registre : [`configs/test_corpora.yaml`](configs/test_corpora.yaml) — chaque entrée définit `data_csv`, `emb_csv`, `display_name`. Défaut : **`metallurgie`**.

**Ajouter un corpus** : `dataset/data_<id>.csv` + entrée dans le registre, puis :

```bash
export TEST_CORPUS=<id>   # ou CORPUS= pour macro_transfer
python scripts/export_corpus_embeddings.py --corpus <id>
# ou : CORPUS=<id> sbatch jobs/export_corpus_embeddings.sh
```

Utilisé par : entraînement SCGM, contrastifs, jobs raw/test emb, notebooks 01 / 05 / 06 / 07 (`TEST_CORPUS`).

### Arborescence `output` vs `output_test`

| Racine | Contenu |
|--------|---------|
| `output/<method>/` | **BTP uniquement** : checkpoints, métriques BTP, exports BTP (sans topics test) |
| `output_test/<corpus>/<method>/` | Métriques / projections **test** (`metrics_geometry_test.csv`, …) |
| `output_test/<corpus>/raw_embedding/` | Embedding brut test |
| `output_test/<corpus>/macro_transfer/<method>/` | Transfert macro + topics BERTopic (`theme_label` via `bertopic.representation` par défaut) |
| `output_test/<corpus>/bn_results/` | Sorties notebook 04 (BN) |

Les sorties vivent sous `output/` et `output_test/` (voir `configs/paths.yaml`).

## Transfert macro FSP + topics cible (corpus test)

Pipeline **`macro_transfer/`** — **Frozen Source Prototypes (FSP)** uniquement : encodeur source **figé** (SCGM, contrastif ou embedding brut Qwen) → prototypes source → assignation cible par distances → **BERTopic** intra-macro (toujours exécuté dans le job nominal).

**Checkpoints source** (BTP) : `output/<encodeur>/checkpoints/` (ex. `softtriple`, `scgm_text`, `supcon`, `batch_triplet`). Pour `raw_embedding`, embeddings pré-calculés : `embeddings/Qwen3-Embedding-0.6B_btp.csv` + export multi-corpus (`jobs/export_corpus_embeddings.sh`).

Paramètres : [`configs/frozen_source_prototypes.yaml`](configs/frozen_source_prototypes.yaml) (**config unique**). Encodeur choisi via `BASE_METHOD` dans `jobs/run_frozen_source_prototypes.sh`.

**Libellés topics** : macros A0–C ([`configs/accident_macros.yaml`](configs/accident_macros.yaml)) + contexte sectoriel ([`configs/corpus_prompt_context.yaml`](configs/corpus_prompt_context.yaml)). Clé OpenAI requise si `bertopic.representation` utilise OpenAI.

| Étape | Commande |
|-------|----------|
| FSP + BERTopic | `BASE_METHOD=scgm_text CORPUS=<id> bash jobs/run_frozen_source_prototypes.sh` |
| Autre encodeur | `BASE_METHOD=softtriple CORPUS=<id> bash jobs/run_frozen_source_prototypes.sh` |
| **SoftTriple centres natifs** | `CORPUS=<id> bash jobs/run_softtriple_native_fsp.sh` (job dédié, sortie `softtriple_native/`) |
| Embedding brut | `BASE_METHOD=raw_embedding CORPUS=<id> bash jobs/run_frozen_source_prototypes.sh` |
| CLI natif | `python scripts/run_softtriple_native_fsp.py --corpus <id>` |
| Notebook 09 | diagnostics SoftTriple natif vs prototype moyen (`build_notebook_09_softtriple_native_fsp.py`) |
| CLI | `python scripts/run_frozen_source_prototypes.py --config configs/frozen_source_prototypes.yaml --base-method softtriple --corpus <id>` |
| Notebook 06 | topics + calibration FSP |
| Notebook 08 | comparaison raw vs encodeur (`FSP_BASE_METHOD`) |

**Sorties** : `output_test/<corpus_id>/macro_transfer/frozen_source_prototypes/<base_method>/`

Pour **SoftTriple**, deux modes coexistent :
- `softtriple/` — job FSP générique (`BASE_METHOD=softtriple`) : prototype moyen des embeddings BTP par macro ;
- `softtriple_native/` — job dédié : affectation via centres appris `W_{r,k}` (`macro_probs_softtriple`, cosine par défaut au transfert). L'entraînement SoftTriple peut utiliser une métrique euclidienne ; le YAML natif permet `distance_metric: euclidean` pour comparer.

Les anciens runs sous `frozen_source_prototypes/scgm` ou `raw` restent lisibles (alias staging BN).

### Fine-tuning supervisé macro (tête softmax)

Distinct du notebook **07** (sklearn sur embeddings Qwen **figés** CSV) : ici Qwen gelé + projecteur ψ (`projection: linear|mlp_sklearn`, défaut `mlp_sklearn`) + tête CE ; avec `cache_backbone_embeddings: true`, Qwen n'est encodé qu'**une fois** (ou lu depuis `embeddings/Qwen3-Embedding-0.6B_btp.csv`), puis seuls ψ et la tête s'entraînent par epoch.

`mlp_sklearn` aligne l'architecture MLP du notebook 07 (`hidden_layer_sizes: [256, 128]`, ReLU) ; le baseline sklearn applique aussi un **StandardScaler** sur h — reproduit via `model.standardize_backbone: true` (ou `STANDARDIZE_BACKBONE=true` dans les jobs).

Rééquilibrage des classes (mutuellement exclusif si les deux activés) :
- `model.oversampling: true` — sur-échantillonnage équilibré sur le train (comme baseline MLP sklearn), défaut `false`
- `model.class_weight: balanced` — pondération CE, défaut `null` (pas de rééquilibrage)

Modes backbone :
- **A gelé + cache** : `backbone_trainable: false` (rapide)
- **B partiel** : `backbone_trainable: true`, `train_last_n_layers: 4` (recommandé si FT)
- **C complet** : `backbone_trainable: true`, `train_last_n_layers: null` (lent ; `gradient_checkpointing` auto si non défini ; `training.use_amp: true` sur GPU)

| Étape | Commande |
|-------|----------|
| Entraînement BTP | `bash jobs/train_supervised_macro_ft.sh` |
| Grid search CV (balanced accuracy) | `MAX_COMBOS=4 SKIP_FINAL_FIT=1 bash jobs/tune_supervised_macro_ft.sh` |
| Grille complète + fit final | `bash jobs/tune_supervised_macro_ft.sh` |

En mode tuning (`cv_only`), seuls les **CSV** sont écrits par combo (`cv/`, `metrics/`) — pas de checkpoints `folds/fold_*` (évite ~1–2 Go × fold × combo sur le disque). Le fit final du meilleur combo écrit `checkpoints/best_model/`.
| Transfert test + BERTopic | `CORPUS=<id> bash jobs/run_supervised_macro_ft_transfer.sh` |
| Classif seule | `RUN_BERTOPIC=false CORPUS=<id> bash jobs/run_supervised_macro_ft_transfer.sh` |
| BERTopic sans juge LLM | `JUDGE_ENABLE=false CORPUS=<id> bash jobs/run_supervised_macro_ft_transfer.sh` |
| Notebook 10 | `python scripts/build_notebook_10_supervised_macro_ft.py` |

Transfert : `run_bertopic` (YAML ou `RUN_BERTOPIC`) active BERTopic ; `judge_enable` (YAML ou `JUDGE_ENABLE`) contrôle les appels LLM du **juge qualité** (les titres topics restent générés via `bertopic.representation`).

Configs : [`configs/methods/supervised_macro_ft.yaml`](configs/methods/supervised_macro_ft.yaml), [`configs/supervised_macro_ft_transfer.yaml`](configs/supervised_macro_ft_transfer.yaml), grille [`configs/tuning/supervised_macro_ft_grid.yaml`](configs/tuning/supervised_macro_ft_grid.yaml).

Sorties : `output/supervised_macro_ft/` (train, `metrics/kfold_geometry_*.csv`, `metrics_geometry_btp.csv`) ; tuning : `output/supervised_macro_ft/tuning/grid_summary.csv`, `best_combo.json` ; `output_test/<corpus>/macro_transfer/supervised_macro_ft/` (`transfer/metrics_geometry.csv`). BERTopic utilise les embeddings **z** projetés.

Sorties principales :
- `transfer/source_prototypes.csv`
- `transfer/target_macro_predictions.csv`
- `transfer/metrics.json` (si labels cible)
- `transfer/bertopic_input_*.csv`
- `topics_bertopic/assignments.csv`, `themes_by_macro.csv`

## Réseaux bayésiens

- `notebooks/04_bayesian_network_macro_transfer.ipynb` — staging depuis **macro_transfer** (corpus test), sortie `output_test/<TEST_CORPUS>/bn_results/` : graphe BN statique (`figures/static/bn_network.png`), interactif Plotly (`figures/interactive/bn_network.html`), scénarios récurrents (`tables/recurring_scenarios.csv`)

Dépendances : `numpy<2`, `pgmpy>=0.1.23,<1.0`. Utiliser le même interpréteur Python que le noyau Jupyter (`import sys; print(sys.executable)`).

## Notebooks

Le **corpus** (BTP, métallurgie, etc.) est défini dans les cellules *Parameters* ou les YAML `configs/methods/`, pas dans le nom du fichier `.ipynb`.

| Notebook | Rôle |
|----------|------|
| `00_check_data.ipynb` | Aperçu du CSV configuré |
| `01_compare_embedding_methods.ipynb` | Comparaison **Embedding brut + Batch Triplet / SupCon / SoftTriple / SCGM** — BTP **K-fold (μ±σ)**, BTP fit final, test métallurgie ; η² / IPR |
| `02_scgm_text_results.ipynb` | **Lecture seule** — BTP (`output/scgm_text`) ; test (`output_test/<TEST_CORPUS>/scgm_text`). Topics : notebook 06 / `run_frozen_source_prototypes.sh`. |
| `04_bayesian_network_macro_transfer.ipynb` | BN sur corpus test — graphe statique/interactif + tableau scénarios récurrents (A0→…→C) |
| `05_view_batch_triplet_results.ipynb` | Résultats Batch Triplet (`output/batch_triplet/`) — métriques + **PCA/t-SNE** BTP et test (macro + centroïdes) si `embeddings/final_embeddings_*.csv` présents |
| `05_view_softtriple_results.ipynb` | Résultats SoftTriple (idem) |
| `05_view_supcon_results.ipynb` | Résultats SupCon (idem) |
| `06_macro_transfer_topics.ipynb` | **Lecture seule** — FSP (probas/distances macro, BERTopic inputs, calibration/erreurs) |
| `07_supervised_macro_baseline.ipynb` | **Exécutable** — classifieurs sklearn sur Qwen brut (GroupKFold BTP → test métallurgie → BERTopic) |
| `10_supervised_macro_ft_results.ipynb` | **Lecture** — fine-tuning CE (θ + tête softmax) ; BERTopic sur h_t adapté (voir jobs dédiés) |
| `08_fsp_macro_transfer_results.ipynb` | **Lecture seule** — diagnostics FSP (raw vs encodeur, confusion/report, distances, BERTopic) |

Entraînement **hors notebook** : `scripts/train_scgm_text.py` ou `jobs/*.sh` (SLURM). Les notebooks chargent checkpoints, `train_log.csv` et exports déjà produits.

**JupyterHub (HPC2)** : JupyterLab ne voit que `~/notebooks`. Créer un lien vers le projet, par ex. `ln -sfn ~/SAFER/text ~/notebooks/SAFER_text`, puis kernel Python avec le venv du projet (`ipykernel install --user --name safer-text`).

Les fichiers `notebooks/*.ipynb` ne sont **pas versionnés** (restent sur la machine / le cluster). Après `git pull`, régénérer :

```bash
python scripts/build_analysis_notebooks.py        # 00, 01_compare, 05_view_*
python scripts/build_notebook_02_scgm_results.py  # 02_scgm_text_results
python scripts/build_notebook_04_bn_macro_transfer.py
python scripts/build_notebook_06_macro_transfer_topics.py
python scripts/build_notebook_07_supervised_macro_baseline.py
python scripts/build_notebook_08_fsp_macro_transfer.py
python scripts/build_notebook_09_softtriple_native_fsp.py
python scripts/build_notebook_10_supervised_macro_ft.py
```

## Métriques principales

- **eta2_macro_balanced**, **eta2_macro_balanced_perc**, **eta2_weighted** — structuration macro
- **IPR_mean** — préservation intra-rôle (notebook 01)

Pas d'Accuracy / F1 / NMI dans le tableau principal de comparaison des méthodes.

## Prompts

Pipeline principal : `text_col=sentence`, `use_prompt: false` dans toutes les configs `configs/methods/`.

## Méthodes contrastives

**Backend** : encodeur HF unifié (`ContrastiveEncoder` = `TextBackbone` + projecteur optionnel) pour **Batch Triplet**, **SoftTriple** et **SupCon**. Le backbone peut être gelé (`backbone_trainable: false`, défaut) avec cache d'embeddings (`cache_backbone_embeddings: true`), partiellement entraîné (`train_last_n_layers`) ou entièrement fine-tuné.

**Projecteur** (`model.use_projector`, `projection: linear | mlp_sklearn`, `hiddim`) : entraîné avec la loss contrastive ; désactivable (`use_projector: false`).

**Post-évaluation classification** (`post_eval:`) : après entraînement contrastif, régression logistique sklearn sur les embeddings encodés (fit train / eval val en CV ; fit BTP 100 % / eval BTP + test en fit final). La **sélection du best checkpoint reste géométrique** (δ_macro = `eta2_macro_balanced_perc`). Métriques : `accuracy`, `macro_f1`, `balanced_accuracy` → `metrics_classification_{btp,test}.csv` et colonnes `val_*` dans `kfold_summary.csv`.

**Métrique principale** : δ_macro (%) = `eta2_macro_balanced_perc` = 100 × η²_macro_balanced (structuration macro de l'espace). Sélection du meilleur checkpoint sur le **val** via δ_macro (plus `eval_loss` pour SoftTriple).

**Losses d'entraînement** (boucle PyTorch HF unifiée) :
- **Batch triplet** : batch-hard triplet sur embeddings + `PKBatchSampler` ; `training.distance_metric` = euclidien par défaut.
- **SupCon** : [HobbitLong/SupContrast](https://github.com/HobbitLong/SupContrast) sur embeddings L2-normalisés ; `training.distance_metric` doit être `cosine` ; hyperparamètres dans `supcon:` (`temperature`, `base_temperature`, `contrast_mode`).
- **SoftTriple** : loss native ; euclidien par défaut. Entraînement custom avec **AMP GPU** (bf16/fp16), val `loss` + géométrie en **une passe** par epoch. Console : `[SoftTriple epoch=k/N] train_loss=… | val_loss=… | …` ; détail dans `metrics/train_log.csv`.

Les métriques val/export (η²) restent calculées sur embeddings L2-normalisés, indépendamment de la loss.

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

**PKBatchSampler (Batch Triplet)** : batches équilibrés auto — P = nombre de macros dans le fold train, K = `batch_size / P` (ex. 64 et 4 classes → 16 ex./classe). Vérification : `python scripts/debug_pk_sampler.py`.

**SupCon** : sampler shuffle standard, pas PK.

**Batch Triplet / SupCon / SoftTriple** : boucle HF unifiée. Pendant l'entraînement : `[BatchTriplet epoch=k/N] train_loss=… | eta2_macro_balanced_perc=…` (idem `SupCon`, `SoftTriple`) ; détail dans `metrics/train_log.csv`.

Checkpoints : `checkpoints/best_model/contrastive_encoder.pt` (+ `softtriple_state.pt` pour SoftTriple).


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

Package : `contrastive_methods/` (`train.py`, `tuning.py`, `encoder_model.py`, `hf_training_common.py`, `post_eval.py`, `training_*.py`, `eval_geometry.py`, `training_log.py`).

## Tests

```bash
python -m pytest tests/
```
