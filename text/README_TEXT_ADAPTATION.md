# SCGM Text Adaptation for BTP Accident Narrative Embeddings

## Objectif scientifique

Cette adaptation applique SCGM-G à des embeddings texte fixes de récits d'accidents du BTP. Les macro-rôles `A0`, `A1`, `B`, `C` jouent le rôle des superclasses SCGM, tandis que les composantes latentes `z` modélisent des thèmes intra-macro non observés.

## Données attendues

- Métadonnées segmentées : [`dataset/data_btp.csv`](dataset/data_btp.csv)
- Embeddings pré-calculés : [`embeddings/`](embeddings/)
- Alignement par `doc_id = index + 1` sur les lignes filtrées de `data_btp.csv`

## Installation

```bash
cd text
pip install -r requirements.txt
```

Le code image historique : [`../images/requirements_legacy_image.txt`](../images/requirements_legacy_image.txt).

## Projections (backbone natif)

Le modèle texte utilise un paramètre **`projection`** : `identity` (pas de couche linéaire, `hiddim` = dimension du backbone), `linear` ou `mlp`. Le mode **`mlp`** (Linear → ReLU → Linear) est le plus proche du `fc_enc` du [SCGM-G officiel](https://github.com/nijingchao/SCGM). Les anciens checkpoints n’avaient que `with_mlp` (bool) : au chargement, `with_mlp=True` → `mlp`, `False` → `linear`.

## Fidelity to official SCGM-G

Cette adaptation reproduit le **cœur probabiliste SCGM-G** (μ_y, μ_z, E-step Sinkhorn, pertes ls1/ls2/ls3), pas le pipeline image complet (ResNet50, BREEDS, augmentations).

**Conservé**

- Centres macro `mu_y` et latents `mu_z`
- `p(z|x)`, `p(y|z)`, marge macro `p(y|x) = sum_z p(z|x) p(y|z)`
- E-step Sinkhorn sur les scores latents (voir ci-dessous)
- Pertes ls1, ls2, ls3
- Self-distillation optionnelle (`DistillKL`, `--use_self_distillation`)

**Adapté au texte**

- Images ResNet → embeddings texte pré-calculés
- `fc_enc` → `projection` (`identity` / `linear` / `mlp`)
- Optimiseur par défaut : AdamW (mode pragmatique) ; SGD + cosine disponible (mode strict)
- Split par `accident_id` (évite la fuite entre segments)
- Évaluation macro / subtype / topics

**Non reproduit exactement**

- Entraînement end-to-end image
- Protocole BREEDS / few-shot image
- Multi-GPU DataParallel officiel

| Aspect | SCGM-G officiel | SAFER SCGM-Text | Statut |
|--------|-----------------|-----------------|--------|
| Input | Images | Embeddings texte | Adapté |
| Backbone | ResNet50 | Projecteur texte | Adapté |
| mu_y / mu_z | oui | oui | Conservé |
| E-step Sinkhorn | oui | oui | Conservé |
| Optimiseur | SGD + cosine | AdamW ou SGD (config) | Configurable |
| Self-distillation | optionnelle (β) | optionnelle | Configurable |
| Évaluation | BREEDS fine labels | macro / subtype / topic | Adapté |

**Note sur l’E-step :** `compute_latent_sinkhorn_scores` renvoie un tenseur `(n, K)` utilisé pour Sinkhorn. Ce n’est **pas** la marge macro `p(y|x)` mais le score `p(z|x) * p(y_obs|z)` (comme dans le code officiel, nommé `batch_prob_y_x`).

### Modes d’entraînement

```bash
cd text

# Mode strict (SGD + cosine, projection mlp) — fidélité aux choix d’optimisation officiels
python scripts/train_scgm_text.py \
  --config configs/scgm_text_strict_fidelity.yaml \
  --run_name scgm_text_strict_qwen06 \
  --scgm_strict_mode

# Mode pragmatique (AdamW, projection linear) — défaut historique adapté au texte
python scripts/train_scgm_text.py \
  --config configs/scgm_text_pragmatic_adamw.yaml \
  --run_name scgm_text_adamw_qwen06 \
  --text_pragmatic_mode
```

Logs détaillés : `{output_dir}/metrics/train_log.csv` et `epoch_metrics.jsonl` (ls1–ls3, Sinkhorn, entropies, NMI subtype si disponible). Le fichier `logs.csv` à la racine du run est conservé pour compatibilité.

Configs : [`configs/scgm_text_default.yaml`](configs/scgm_text_default.yaml), [`configs/scgm_text_strict_fidelity.yaml`](configs/scgm_text_strict_fidelity.yaml), [`configs/scgm_text_pragmatic_adamw.yaml`](configs/scgm_text_pragmatic_adamw.yaml).

## Entraînement

```bash
cd text
python scripts/train_scgm_text.py \
  --data_csv dataset/data_btp.csv \
  --emb_csv embeddings/Qwen3-Embedding-0.6B_btp.csv \
  --output_dir runs/scgm_text_qwen06 \
  --batch_size 512 \
  --epochs 100 \
  --hiddim 128 \
  --n_class 4 \
  --n_subclass 32 \
  --tau 0.1 \
  --alpha 0.5 \
  --lmd 25 \
  --n_iter_estep 5 \
  --val_ratio 0.1 \
  --group_col accident_id \
  --seed 42 \
  --device cuda \
  --projection identity
```

Options utiles : `--optimizer adamw|sgd`, `--scheduler none|cosine`, `--use_self_distillation`, `--resume_from_checkpoint`, `--run_name` (sortie sous `runs/scgm_text/<run_name>/`).

## Export des outputs

```bash
cd text
python scripts/export_scgm_text_outputs.py \
  --data_csv dataset/data_btp.csv \
  --emb_csv embeddings/Qwen3-Embedding-0.6B_btp.csv \
  --checkpoint runs/scgm_text_qwen06/best_model.pt \
  --output_dir runs/scgm_text_qwen06/exports
```

## Thèmes enrichis (OpenAI)

Après export, le fichier `themes_by_z.csv` peut être enrichi via [`scgm_text/openai_theme_labels.py`](scgm_text/openai_theme_labels.py) : le modèle reçoit jusqu’à **N extraits** de segments (découpe de `top_sentences` au séparateur ` || `) et produit notamment une colonne **`theme_summary`** : **étiquette de topic** (consigne **6 à 10 mots**), contrainte côté code pour la carte 2D et l’exploration.

```bash
set OPENAI_API_KEY=...
python -m scgm_text.openai_theme_labels runs/scgm_text_qwen06/exports/themes_by_z.csv --n-example-texts 5
```

Options utiles : `--n-example-texts` (défaut 5), `--summary-words-min` (défaut 6), `--summary-words-max` (défaut 10).

Sortie par défaut : `themes_by_z_openai.csv` dans le même dossier. Variable optionnelle : **`OPENAI_BASE_URL`** (Azure ou proxy). Ne commitez jamais la clé ; placez-la dans un fichier ``.env`` à la racine du dépôt ou dans ``scgm_text/.env`` (format ``OPENAI_API_KEY=sk-...``). Le module charge ces fichiers automatiquement si ``python-dotenv`` est installé ; sinon exportez la variable dans le shell.

## Évaluation

```bash
cd text
python scripts/evaluate_scgm_text.py \
  --exports_dir runs/scgm_text_qwen06/exports \
  --output_dir runs/scgm_text_qwen06/evaluation
```

## Interprétation des sorties

- `mu_y.npy` : ancres macro A0/A1/B/C
- `mu_z.npy` : centres latents
- `prob_y_x.npy`, `prob_z_x.npy`, `prob_y_z.npy` : probabilités hiérarchiques
- `z_assignments.csv` : assignation latente par segment
- `themes_by_z.csv` et `themes_by_macro_z.csv` : thèmes locaux par composante
- `themes_by_z_openai.csv` : enrichissement OpenAI (`theme_title`, **`theme_summary`** = libellé topic 6–10 mots, `theme_keywords`)

## Notebook expérimental

Le notebook [`notebooks/01_scgm_text_btp_experiment.ipynb`](notebooks/01_scgm_text_btp_experiment.ipynb) orchestre le pipeline texte cellule par cellule : exploration des données, split par `accident_id`, entraînement via `scripts/train_scgm_text.py`, export, évaluation, **enrichissement OpenAI optionnel**, **carte 2D UMAP + DataMapPlot** (`figures/datamap_segments.png`) avec libellés **`theme_summary`** si `DATAMAP_LABEL_MODE="theme_summary"` et le CSV OpenAI est présent, et figures sous `runs/scgm_text_qwen06_notebook/`.

```bash
cd text
pip install -r requirements.txt
python -m ipykernel install --user --name scgm-text
jupyter notebook notebooks/01_scgm_text_btp_experiment.ipynb
```

Sorties attendues : `figures/`, `tables/`, `exports/`, `evaluation/` et `notebook_summary.json` dans le dossier `OUTPUT_DIR` du notebook.

## Limites

- Les labels macro proviennent de `pred_label`, pas d'une annotation experte consolidée.
- `pred_subtype` est réservé au diagnostic exploratoire.
- Les embeddings 4B et 8B sont volontairement hors de la première validation.
- Coût et non-déterminisme des appels OpenAI (`temperature`).

## Prochaines étapes vers MALT

- Transfert inter-corpus
- Fine-tuning de l'encodeur
- Couplage avec des contraintes structurelles plus riches sur les scénarios d'accident
