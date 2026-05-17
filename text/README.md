# Pipeline SAFER — texte

Analyse de récits d'accidents : SCGM-G sur embeddings BTP, transfert macro-ancré (MALT), méthodes contrastives, comparaison de topics et réseaux bayésiens exploratoires.

## Installation

```bash
cd text
pip install -r requirements.txt
```

Variables d'environnement : `HF_TOKEN` ou `HUGGING_FACE_HUB_TOKEN` dans `.env` (modèles Hugging Face). `OPENAI_API_KEY` optionnel (enrichissement de thèmes). Ne jamais committer `.env`.

## Organisation

| Dossier | Rôle |
|---------|------|
| `dataset/` | CSV métadonnées BTP / métallurgie |
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
| `notebooks/` | Analyse et pipelines expérimentaux |
| `legacy/` | Code historique (contrastif v0, anciens jobs) |
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
python scripts/train_scgm_text.py --config configs/methods/scgm_text.yaml
python scripts/export_scgm_text_outputs.py \
  --checkpoint resultats/scgm_text/checkpoints/best_model.pt \
  --output_dir resultats/scgm_text
python scripts/evaluate_scgm_text.py \
  --exports_dir resultats/scgm_text/embeddings \
  --output_dir resultats/scgm_text/metrics

# 3. MALT (BTP → métallurgie)
python scripts/train_malt_target.py --config configs/malt_btp_to_mettalurgie_qwen06.yaml
python scripts/export_malt_outputs.py \
  --checkpoint resultats/malt/best_model.pt \
  --source_checkpoint resultats/scgm_text/checkpoints/best_model.pt \
  --output_dir resultats/malt/exports

# 4. Méthodes contrastives (legacy + post-traitement automatique)
python scripts/train_batch_triplet.py   # → resultats/batch_triplet/embeddings/ + metrics/
python scripts/train_softtriple.py
python scripts/train_supcon.py
# Post-traitement manuel si besoin :
python scripts/postprocess_contrastive_results.py --method batch_triplet

# 5. Agrégation
python scripts/collect_results.py
python scripts/compare_methods.py
```

### Jobs SLURM

```bash
cd jobs
sbatch train_scgm_text.slurm
sbatch train_batch_triplet.slurm
# … ou : bash submit_all.sh
sbatch compare_methods.slurm
```

Logs : `resultats/<method>/logs/slurm-*.out`. Cache HF : `$SCRATCH/hf_cache` si défini.

`submit_all.sh` crée les dossiers `resultats/*/logs/` **avant** `sbatch` (évite les échecs SLURM si `resultats/` n'existe pas encore). Les fichiers `*.slurm` utilisent des fins de ligne LF (voir `.gitattributes`).

## SCGM-Text

Macros observées `A0`–`C` ; latents `z` = thèmes intra-macro. Données : `dataset/data_btp.csv` + `embeddings/Qwen3-Embedding-0.6B_btp.csv` (alignement `doc_id`).

**`input_mode`** : `text` (backbone HF fine-tunable) ou `precomputed_embeddings` (colonnes `dim_*`). **`projection`** : `identity` | `linear` | `mlp`. Avec `text` + `freeze_backbone=false`, `identity` signifie **h = f_θ(x)** (pas des embeddings figés).

**Sélection du meilleur checkpoint** (`best_model.pt`) : `val_eta2_macro_balanced` par défaut (pas F1). Option `--best_checkpoint_metric composite` avec `--best_checkpoint_lambda` (score = eta² − λ·C1). Diagnostics classifieur / subtype : `--compute_classifier_diagnostics` / `--compute_subtype_diagnostics` (désactivés par défaut).

Presets d'entraînement :

```bash
python scripts/train_scgm_text.py --config configs/methods/scgm_text.yaml
python scripts/train_scgm_text.py --config configs/scgm_text_strict_finetune_identity.yaml --strict_finetune_identity
```

Thèmes OpenAI (optionnel) :

```bash
python -m scgm_text.openai_theme_labels resultats/scgm_text/topics/themes_by_z.csv --n-example-texts 5
```

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
| `01_scgm_text_experiment.ipynb` | Pipeline SCGM complet (expérimental) |
| `02_malt_btp_to_mettalurgie_transfer.ipynb` | Transfert MALT (ex. BTP → métallurgie) |
| `03_compare_malt_bertopic_kmeans_topics.ipynb` | Qualité topics |
| `04_malt_to_bayesian_network.ipynb` | BN depuis MALT |
| `04_bayesian_network_from_scgm.ipynb` | BN depuis SCGM |
| `05_view_batch_triplet_results.ipynb` | Résultats Batch Triplet (`resultats/batch_triplet/`) |
| `05_view_softtriple_results.ipynb` | Résultats SoftTriple |
| `05_view_supcon_results.ipynb` | Résultats SupCon |

`01_draft.ipynb` : brouillon obsolète — ne pas utiliser.

Régénération :

```bash
python scripts/build_analysis_notebooks.py   # 00, 01_compare, 05_view_*
python scripts/rebuild_notebook_01.py
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

## Legacy

Code historique sous `legacy/` (contrastif v0, anciens scripts d'export). Les scripts `train_*` contrastifs délèguent au legacy puis exécutent automatiquement `postprocess_contrastive_results.py` (embeddings → `resultats/<method>/embeddings/final_embeddings.csv`, `metrics/metrics_geometry.csv`, `configs/config_resolved.yaml`).

```bash
python scripts/train_batch_triplet.py
```

## Tests

```bash
python -m pytest tests/
```
