# Scénarios récurrents d'accidents

Protocole dédié (indépendant de SCGM / contrastifs / BERTopic) :

**unités annotées + embeddings Qwen figés → UMAP–HDBSCAN par rôle → \(S_R\) / \(D_U\) → sélection \(\arg\max S_R\) → seed sensitivity → labels (notebook) → BN latent (\(Z\)) → MPE → `recurrent_scenarios.csv`.**

Corpus enregistrés : `btp`, `caou`, `metallurgie`.

## Workflow

| Étape | Outil | Rôle |
|-------|--------|------|
| 1. Discovery | `run_theme_discovery.py` (+ job Slurm) | Grille UMAP–HDBSCAN, \(D_U\), \(S_R\), sélection auto \(\arg\max S_R\) (tie-break \(D_U\)), sensibilité multi-seeds. |
| 2. Résultats / thèmes | `notebooks/topic_modeling_results_{corpus}.ipynb` | Affiche les artefacts figés, dictionnaire c-TF-IDF, labels LLM (sans rechoisir la config). |
| 3. BN + scénarios | `notebooks/recurrent_scenarios_bn_analysis_{corpus}.ipynb` | Matrice accident × thèmes, Structural EM avec \(Z\), MPE. |

Un notebook résultats est généré **par corpus** (`caou`, `btp`, `metallurgie`), comme pour le BN.
`topic_modeling_results.ipynb` est un alias du notebook `caou`.

Régénérer les notebooks :

```bash
python recurrent_scenarios/build_theme_discovery_notebook.py
python recurrent_scenarios/build_recurrent_scenarios_results_notebook.py
python recurrent_scenarios/build_bn_analysis_notebook.py
```

## Discovery (job / CLI)

`config.yaml` est la source de vérité. Section `validation:` : resampling, sélection \(S_R\), tie-break \(D_U\), `seed_sensitivity`.

Depuis `text/` :

```powershell
python recurrent_scenarios/run_theme_discovery.py --config recurrent_scenarios/config.yaml --dataset caou --reestimate
```

Sur Slurm :

```bash
DATASET=caou REESTIMATE=1 sbatch jobs/run_recurrent_scenarios_theme_discovery.sh

DATASET=caou STAGE=metrics REESTIMATE=1 sbatch jobs/run_recurrent_scenarios_theme_discovery.sh

DATASET=caou STAGE=select RUN_DIR=recurrent_scenarios/runs/theme_discovery_audit/caou \
  sbatch jobs/run_recurrent_scenarios_theme_discovery.sh

DATASET=caou STAGE=seed RUN_DIR=recurrent_scenarios/runs/theme_discovery_audit/caou \
  sbatch jobs/run_recurrent_scenarios_theme_discovery.sh
```

Stages CLI : `all` | `metrics` | `select` | `seed`.

Sortie par défaut : `runs/theme_discovery_audit/{dataset_id}/`.

## Contrat d'entrée

Table d'unités : `accident_id`, `fact_id`, `sentence`, `pred_label`, `pred_ok` (rôles `A0`, `A1`, `B`, `C`).

Embeddings : `doc_id` aligné sur `fact_id` (ou même ordre) + colonnes `dim_*`.

Un zéro dans la matrice accident × thèmes signifie « thème non observé dans les unités disponibles », pas une preuve d'absence.

## Choix de modélisation

- Clustering indépendant par rôle sur embeddings figés L2-normalisés.
- Critère de sélection : \(S_R\) (moyenne non pondérée des \(S_{ck}\) ; \(S_{ck}\) = moyenne des Jaccard best-match sur réplicats où le facteur est observable). \(O_{ck}=B_{ck}/B\) est un diagnostic séparé.
- \(D_U\) (DBCV dans l'espace UMAP) : diagnostique + tie-break si égalité de \(S_R\).
- Sensibilité multi-seeds UMAP après sélection (ne change pas \(c_r^\star\)).
- BN : mélange contraint A0→A1→B→C avec famille latente \(Z\) (pas de preuve causale).

## Sorties principales du job

- `audit_input_summary.csv`, `config_resolved.yaml`, `parallel_runtime.json`
- `selected_configurations.csv` — une config par rôle
- `discovery/<role>/candidate_metrics.csv`, `stability_summary.csv`, `stability_theme.csv`
- `discovery/<role>/candidate_partitions/*.npy`
- `discovery/<role>/selected/` — partition figée
- `discovery/<role>/seed_sensitivity/` — tables + figure
- `figures/stability_landscape_all_roles.png`, `factor_resampling_<role>.png`

Après notebook résultats : `topics_manual/…`

Après notebook BN : `bayesian_networks/recurrent_scenarios.csv` (+ supports, prototypes, figures)

## Hors scope

Transfert inter-secteur et pipeline BERTopic + `bn_pipeline` (autre notion de scénario).
