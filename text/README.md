# Pipeline SAFER — texte

Analyse de récits d'accidents : SCGM-G sur embeddings BTP, méthodes contrastives et baselines supervisées.

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
| `output/raw_embedding/`, `batch_triplet/`, `softtriple/`, `supcon/` | Méthodes entraînées + éval classification (BTP + OOD) |

## Workflow principal

```bash
cd text

# 1. Embedding brut BTP + test (classification)
sbatch jobs/export_raw_embeddings_eval.sh
# ou : python scripts/export_raw_embeddings.py

# 2. SCGM BTP (+ eval test → output_test/ en fin de train)
sbatch jobs/train_scgm_text.sh
sbatch jobs/export_corpus_embeddings.sh   # embeddings Qwen (btp + metallurgie + caou)
# Sorties clés : output/<method>/metrics/metrics_classification_*.csv, cross_domain_generalization.csv,
#   embeddings/projected_{btp,metallurgie,caou}.npy,
#   predictions/predictions_{btp,metallurgie,caou}.csv (+ transfer/target_macro_predictions.csv)
#   embeddings/Qwen3-Embedding-0.6B_<corpus>.csv

# 3. Méthodes contrastives (entraînement natif, YAML configs/methods/*.yaml)
python scripts/train_batch_triplet.py   # → output/<method>/embeddings/final_embeddings.csv
python scripts/train_softtriple.py
python scripts/train_supcon.py
# Recalcul métriques uniquement si besoin :
# Puis fit final 100 % BTP → metrics_classification_*.csv + cross_domain_generalization.csv

# 4. Embeddings encodeur figé (Qwen) — job ou CLI :
sbatch jobs/export_corpus_embeddings.sh
# CORPUS=metallurgie sbatch jobs/export_corpus_embeddings.sh
# python scripts/export_corpus_embeddings.py --config configs/export_embeddings.yaml --corpus metallurgie

# 5. Tuning K-fold (sélection sur mean val_balanced_accuracy)
cd jobs && bash submit_tuning_all.sh
# ou : sbatch tune_scgm_text.sh tune_softtriple.sh tune_supcon.sh tune_batch_triplet.sh
# Variables : TEST_CORPUS, MAX_COMBOS, SKIP_FINAL_FIT, SEED, GRID_CONFIG
# Sorties : output/<method>/tuning/grid_summary.csv, best_combo.json
# Puis fit final 100 % BTP → metrics_classification_*.csv + cross_domain_generalization.csv

# 4. Notebooks de lecture (résultats par méthode)
python scripts/build_analysis_notebooks.py        # 00 + 05_view_*
python scripts/build_notebook_02_scgm_results.py  # SCGM
python scripts/build_notebook_06_bn_macro_constrained.py
python scripts/build_notebook_07_supervised_macro_baseline.py
```

### Évaluation BTP vs test

| Corpus | Chemin | Encodeur |
|--------|--------|----------|
| BTP (entraînement) | `dataset/data_btp.csv` | Best model fine-tuné (contrastifs) ou tête SCGM + `embeddings/Qwen3-Embedding-0.6B_btp.csv` |
| Métallurgie (test) | `dataset/data_metallurgie.csv` | `embeddings/Qwen3-Embedding-0.6B_metallurgie.csv` ; métriques raw : `output_test/metallurgie/raw_embedding/` |
| Caou (test) | `dataset/data_caou.csv` | `embeddings/Qwen3-Embedding-0.6B_caou.csv` |

**Train simple** (`jobs/train_*.sh`, `n_folds: 5`) : (1) K-fold → `cv/cv_summary.csv` + `kfold_summary.csv` (CV classification LR : `val_balanced_accuracy`, etc.) ; (2) **fit final 100 % BTP** → `checkpoints/best_model` (sélection **train_loss**) ; (3) eval **BTP + métallurgie + caou** → `metrics_classification_*.csv`, `cross_domain_generalization.csv`, embeddings `projected_*.npy`, prédictions ligne à ligne `predictions/predictions_<corpus>.csv`.

**Tuning** (`jobs/tune_*.sh`) : même K-fold par combo → `grid_summary.csv` reprend les colonnes `mean_*` / `std_*` (géométrie + temps fold) ; le fit final n’est pas rejoué par combo (sélection puis fit final du meilleur combo).

### Jobs SLURM

```bash
cd jobs
sbatch export_raw_embeddings_eval.sh  # métriques embedding brut BTP + test
sbatch train_scgm_text.sh             # inclut eval BTP + OOD
sbatch export_corpus_embeddings.sh    # CSV Qwen si absents
sbatch train_batch_triplet.sh
# … ou : bash submit_all.sh
```

Logs SLURM : `jobs/slurm-<job_name>-<job_id>.out` (et `.err`) après `sbatch` depuis `jobs/`. Cache HF : `$SCRATCH/hf_cache` si défini. Jobs GPU : `--constraint='a100|h100'`, `--mem=64G`. Les scripts `jobs/*.sh` utilisent des fins de ligne LF (voir `.gitattributes`).

## SCGM-Text

Macros observées `A0`–`C` ; latents `z` = thèmes intra-macro. Données : `dataset/data_btp.csv` + `embeddings/Qwen3-Embedding-0.6B_btp.csv` (alignement `doc_id`).

Pipeline **end2end** : texte → Qwen (`backbone_trainable` / `train_last_n_layers` dans la config) → projecteur (`linear` | `mlp`) → tête SCGM.

**Config unique** : [`configs/methods/scgm_text.yaml`](configs/methods/scgm_text.yaml) — modes backbone documentés en tête du fichier (gelé / k dernières couches / complet).

**Sélection du meilleur checkpoint** : **train_loss** minimal (contrastifs, SCGM). CV / tuning : `balanced_accuracy` (agrégation post-hoc LR sur embeddings).

```bash
python scripts/train_scgm_text.py --config configs/methods/scgm_text.yaml
# ou
sbatch jobs/train_scgm_text.sh
```

**Topics BERTopic** : section homogène dans les notebooks **05_view** et **08** (config `configs/bertopic_notebook.yaml` + `configs/bertopic_macro_shared.yaml`). Sorties sous `{RESULTS_DIR}/bertopic_notebook/<corpus>/` (assignments, thèmes LLM, prédictions macro pour le BN). Prérequis représentation LLM : `OPENAI_API_KEY`.

**Réseau bayésien macro-contraint** : notebook **06** (`scripts/build_notebook_06_bn_macro_constrained.py`) consomme ces exports via `bn_pipeline.staging_macro_transfer.stage_bn_exports_from_bertopic_run` — pas d'arcs entre classes macro différentes (ordre A0 → A1 → B → C).

## Corpus de test (configurable)

Registre : [`configs/test_corpora.yaml`](configs/test_corpora.yaml) — chaque entrée définit `data_csv`, `emb_csv`, `display_name`. Défaut : **`metallurgie`**.

**Ajouter un corpus** : `dataset/data_<id>.csv` + entrée dans le registre, puis :

```bash
export TEST_CORPUS=<id>   # ou CORPUS= pour macro_transfer
python scripts/export_corpus_embeddings.py --corpus <id>
# ou : CORPUS=<id> sbatch jobs/export_corpus_embeddings.sh
```

Utilisé par : entraînement SCGM, contrastifs, jobs raw/test emb, notebooks 05 / 07 (`TEST_CORPUS` / `TEST_CORPORA`).

### Arborescence `output` vs `output_test`

| Racine | Contenu |
|--------|---------|
| `output/<method>/` | **BTP uniquement** : checkpoints, métriques BTP, exports BTP (sans topics test) |
| `output/<method>/metrics/` | Classification BTP + OOD (`metrics_classification_*.csv`, `cross_domain_generalization.csv`) |
| `output/<method>/predictions/` | Prédictions ligne à ligne (`predictions_<corpus>.csv` : `pred_macro`, `prob_*`, `confidence`, …) |
| `output_test/<corpus>/raw_embedding/` | Embedding brut test (legacy) |

Les sorties vivent sous `output/` et `output_test/` (voir `configs/paths.yaml`).

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

Override corpus test : `TEST_CORPORA=metallurgie,caou bash jobs/train_supervised_macro_ft.sh`

Configs : [`configs/methods/supervised_macro_ft.yaml`](configs/methods/supervised_macro_ft.yaml), grille [`configs/tuning/supervised_macro_ft_grid.yaml`](configs/tuning/supervised_macro_ft_grid.yaml).

Sorties : `output/supervised_macro_ft/` — CV BTP (`cv/`), fit final (`checkpoints/best_model/`), eval OOD (`metrics/metrics_classification_test_<corpus>.csv`, `metrics/all_test_corpora_metrics.csv`, `metrics/cross_domain_generalization.csv` avec `ba_ood_avg` / `ba_ood_worst`), prédictions CE `predictions/predictions_<corpus>.csv` ; tuning : `output/supervised_macro_ft/tuning/grid_summary.csv`, `best_combo.json`.

## Notebooks

Le **corpus** (BTP, métallurgie, etc.) est défini dans les cellules *Parameters* ou les YAML `configs/methods/`, pas dans le nom du fichier `.ipynb`.

| Notebook | Rôle |
|----------|------|
| `00_check_data.ipynb` | Aperçu du CSV configuré |
| `02_scgm_text_results.ipynb` | **Lecture seule** — SCGM BTP + OOD (`output/scgm_text`) : métriques classification + t-SNE sur `projected_*.npy` |
| `05_view_batch_triplet_results.ipynb` | **Lecture seule** — Batch Triplet : tableau métriques unifié (CV + BTP + OOD), PCA/t-SNE global (`RESULTS_DIR` configurable) |
| `05_view_softtriple_results.ipynb` | **Lecture seule** — SoftTriple (idem + centres effectifs) |
| `05_view_supcon_results.ipynb` | **Lecture seule** — SupCon (idem) |
| `06_bn_macro_constrained.ipynb` | **Exécutable** — BN pgmpy sur topics BERTopic intra-macro (contraintes A0→C), entrée `bertopic_notebook/<corpus>/` |
| `07_supervised_macro_baseline.ipynb` | **Exécutable** — classifieurs sklearn (LR, RF, XGBoost, MLP) sur Qwen brut : CV GroupKFold BTP → eval OOD |
| `08_view_supervised_macro_ft_results.ipynb` | **Lecture seule** — fine-tuning CE macro_ft : tableau métriques unifié, courbes, PCA/t-SNE global, vraie vs prédite (`RESULTS_DIR` configurable) |
| `09_geometry_comparison.ipynb` | **Exécutable** — comparaison η² macro balanced (%) entre méthodes : Qwen brut + runs projetés (`METHOD_SPECS`), tableau global + barplots par corpus |

Les notebooks `05_view_*` et `08_view_*` acceptent un dossier de run custom via `RESULTS_DIR` en tête de notebook (run standard, combo tuning, chemin absolu). Le notebook **09** configure un dossier par méthode via `METHOD_SPECS` (`kind`: `raw` | `projected`).

Entraînement **hors notebook** : `scripts/train_scgm_text.py` ou `jobs/*.sh` (SLURM). Les notebooks chargent checkpoints, `train_log.csv` et exports déjà produits.

**JupyterHub (HPC2)** : JupyterLab ne voit que `~/notebooks`. Créer un lien vers le projet, par ex. `ln -sfn ~/SAFER/text ~/notebooks/SAFER_text`, puis kernel Python avec le venv du projet (`ipykernel install --user --name safer-text`).

Les fichiers `notebooks/*.ipynb` ne sont **pas versionnés** (restent sur la machine / le cluster). Après `git pull`, régénérer :

```bash
python scripts/build_analysis_notebooks.py        # 00 + 05_view_*
python scripts/build_notebook_02_scgm_results.py  # 02_scgm_text_results
python scripts/build_notebook_06_bn_macro_constrained.py  # 06 BN macro-contraint
python scripts/build_notebook_07_supervised_macro_baseline.py
python scripts/build_notebook_08_supervised_macro_ft_results.py  # 08 macro_ft viz
python scripts/build_notebook_09_geometry_comparison.py  # 09 comparaison η²
```

## Métriques principales

- **accuracy**, **macro_f1**, **balanced_accuracy** — classification LR sur embeddings projetés
- **eta2_macro_balanced_perc** — séparation géométrique macro (η² balanced, %) sur embeddings ; notebook **09** (`METHOD_SPECS`)
- **ba_ood_avg**, **ba_ood_worst** — agrégation cross-domain (`cross_domain_generalization.csv`)
- **val_balanced_accuracy** — sélection checkpoint / tuning (K-fold)

## Prompts

Pipeline principal : `text_col=sentence`, `use_prompt: false` dans toutes les configs `configs/methods/`.

## Méthodes contrastives

**Backend** : encodeur HF unifié (`ContrastiveEncoder` = `TextBackbone` + projecteur optionnel) pour **Batch Triplet**, **SoftTriple** et **SupCon**. Le backbone peut être gelé (`backbone_trainable: false`, défaut) avec cache d'embeddings (`cache_backbone_embeddings: true`), partiellement entraîné (`train_last_n_layers`) ou entièrement fine-tuné.

**Projecteur** (`model.use_projector`, `projection: linear | mlp_sklearn`, `hiddim`) : entraîné avec la loss contrastive ; désactivable (`use_projector: false`).

**Post-évaluation classification** (`post_eval:`) : LR sklearn sur embeddings projetés. **Sélection checkpoint** : `train_loss` minimal pendant l'entraînement ; agrégation CV/tuning sur `val_balanced_accuracy`. Sorties : `metrics_classification_*.csv`, `cross_domain_generalization.csv`, `embeddings/projected_*.npy`, `predictions/predictions_<corpus>.csv`.

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

**Batch Triplet / SupCon / SoftTriple** : boucle HF unifiée. Pendant l'entraînement : `[BatchTriplet epoch=k/N] train_loss=…` (SoftTriple : aussi `val_loss`) ; détail dans `metrics/train_log.csv`.

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

**Log d'entraînement** : `metrics/train_log.csv` — `epoch`, `train_loss`, `val_loss` (SoftTriple). Jobs SLURM : cache HF via `HF_HOME` uniquement (`jobs/_env.sh`, pas `TRANSFORMERS_CACHE`).

Sorties tuning : `output/<method>/tuning/grid_summary.csv`, `best_combo.json`, `combos/<combo_id>/`.  
Après tuning : réentraînement sur tout le corpus → `output/<method>/embeddings/final_embeddings.csv`.

Package : `contrastive_methods/` (`train.py`, `tuning.py`, `encoder_model.py`, `hf_training_common.py`, `post_eval.py`, `eval_corpus.py`, `training_*.py`, `training_log.py`). Module partagé : `safer_core/classification_eval.py`.

## Tests

```bash
python -m pytest tests/
```
