# Recurrent accident scenarios

This workspace implements the protocol in
`protocole_scenarios_recurrents_accidents(1) (1).pdf`.

The recommended workflow is split into an execution job and a results notebook:

1. `run_theme_discovery.py` discovers stable themes on Slurm;
2. `notebooks/recurrent_scenarios_results.ipynb` presents the saved results.

The implementation covers:

1. audit and preparation of fact units;
2. fixed, L2-normalized multilingual embeddings;
3. independent A0/A1/B/C UMAP-HDBSCAN clustering with frozen embeddings;
4. UMAP-space DBCV `D_U` as the primary density-validity metric;
5. accident-level clusterwise Jaccard stability `S_R`;
6. Pareto frontier based only on `D_U` and resampling stability;
7. frozen topic dictionary and binary accident-topic matrix;
8. descriptive frequencies, co-occurrence and lift;
9. constrained Bayesian network without `Z`;
10. constrained latent-family mixture with `Z`;
11. grouped out-of-sample log-likelihood, bootstrap arc stability and supported scenarios.

Inter-sector transfer, sector-comparison figures and alternative methods are intentionally
not implemented here.

## Interactive review

The production computation is launched with the Slurm job below. The YAML file is
the single source of truth for paths, UMAP, HDBSCAN, DBCV, resampling and runtime
settings. A short discovery notebook remains available for interactive audits, but
it is not the production launcher.

The principal plot is `figures/pareto_validation_all_roles.png`: grey points are all
candidates, orange diamonds are Pareto non-dominated candidates and the red star is
the selected partition medoid.
`D_U` and `S_R` are not collapsed into a weighted score. Coverage, number of
clusters and accident support remain diagnostics.

After the Slurm job finishes, set `SAFER_THEME_RUN_DIR` to the produced run
directory and open `notebooks/recurrent_scenarios_results.ipynb`. It only loads
CSV/JSON/PNG artifacts and does not rerun UMAP/HDBSCAN.

Discovery outputs are cached in the run directory. With `REESTIMATE = False`, saved
candidate labels, DBCV, stability and Pareto tables are reused. Set it to `True` when
a discovery parameter, grid or resampling design has changed.

## Partial execution

The launcher exposes three independent stages:

- `all`: compute/reuse metrics, select the Pareto partitions and materialize the
  selected topic assignments and topic dictionary;
- `metrics`: compute or reuse only candidate DBCV and accident-level stability
  artifacts. It does not run Pareto selection;
- `pareto`: read the existing `candidate_metrics.csv` and `stability_summary.csv`
  files from a run, then regenerate only Pareto tables, selected labels and figures.
  It does not rerun embeddings, UMAP or HDBSCAN and requires `--run-dir`.

From `text/`, metrics can be launched independently:

```bash
python recurrent_scenarios/run_theme_discovery.py \
  --config recurrent_scenarios/config.yaml --dataset btp \
  --stage metrics --reestimate
```

Then Pareto selection can be regenerated from the same run directory:

```bash
python recurrent_scenarios/run_theme_discovery.py \
  --config recurrent_scenarios/config.yaml --dataset btp \
  --stage pareto --run-dir recurrent_scenarios/runs/audit_btp
```

The equivalent Slurm controls are `STAGE=metrics` or `STAGE=pareto` and
`RUN_DIR=/path/to/existing/run` for the Pareto-only stage. This makes it possible
to change the Pareto/selection logic and regenerate results without spending
another job on metric computation.

The corpus is selected with the CLI option `--dataset` or the `DATASET` environment
variable in the Slurm job. The registered choices are `btp`, `metallurgie`, `caou` and `nicollin`.
Each choice uses its corresponding annotated unit table and frozen Qwen embedding export,
and writes to a separate `runs/audit_<dataset_id>/` directory.

## Command-line execution

From `text/`:

```powershell
python recurrent_scenarios/run_theme_discovery.py --config recurrent_scenarios/config.yaml --dataset btp --reestimate
python recurrent_scenarios/run_theme_discovery.py --config recurrent_scenarios/config.yaml --dataset metallurgie --debug
```

On Slurm, from `text/`:

```bash
DATASET=btp REESTIMATE=1 sbatch jobs/run_recurrent_scenarios_theme_discovery.sh

# Metrics/stability only
DATASET=btp STAGE=metrics REESTIMATE=1 sbatch jobs/run_recurrent_scenarios_theme_discovery.sh

# Pareto only, using an existing metrics run
DATASET=btp STAGE=pareto RUN_DIR=recurrent_scenarios/runs/audit_btp \
  sbatch jobs/run_recurrent_scenarios_theme_discovery.sh
```

With `parallel.n_workers: auto`, the job uses exactly
`max(1, SLURM_CPUS_PER_TASK - 1)` outer workers. Each UMAP fit remains at
`n_jobs: 1`, and BLAS/OpenMP thread counts are pinned to one in the job to avoid
nested oversubscription. Set `parallel.enabled: false` for a sequential audit.

The discovery launcher writes a new run directory under `runs/` and refuses to overwrite it
unless `runtime.overwrite: true` is set. It records the resolved YAML, input summary,
parallel-runtime metadata, candidate partitions, diagnostics, figures and CSV tables.

## Input contract

The default files are the existing BTP annotated units and the existing fixed Qwen embeddings.
The unit table must contain `accident_id`, `fact_id`, `sentence`, `pred_label` and `pred_ok`.
Valid roles are `A0`, `A1`, `B` and `C`. The embedding table must contain `doc_id` matching
`fact_id` or the same row order as the unit table, plus numeric `dim_*` columns.

The code treats a zero in the accident-topic matrix as “topic not observed in the available
units”, not as proof that the factor was absent from the accident.

## Important modelling choice

Each bootstrap clustering applies UMAP directly to the fixed embeddings and fits HDBSCAN
directly in the reduced space. The topic dictionary then uses a transparent c-TF-IDF-inspired
representation on the final consensus assignments. The vocabulary combines the general
stopwords in `config.yaml` with the business stopwords in `stop_metier.txt`; optional additions
can be placed in `topics.additional_stopwords`. The custom part is the accident-level
consensus across UMAP-HDBSCAN repetitions.

The `Z` model is implemented as a mixture of constrained binary Bayesian networks sharing
the same accident-process DAG. `Z` changes the component probabilities and represents latent
families; the process arcs remain directly comparable with the no-`Z` reference. This is
explicitly reported in the outputs and is not presented as causal evidence.

## Main outputs

- `audit_input_summary.csv`: input counts, missingness and role coverage;
- `pareto/<role>/candidate_metrics.csv`: full-data candidate metrics, including `D_U`;
- `pareto/<role>/stability_theme.csv`: best-match Jaccard values by theme and repetition;
- `pareto/<role>/pareto_selection_table.csv`: Pareto flags and diagnostic metrics;
- `pareto/<role>/pareto_partition_agreement.csv`: agreement matrix among Pareto partitions;
- `pareto/<role>/selected_labels.npy`: frozen medoid partition labels;
- `clustering/<role>/topic_assignments.csv`: frozen role-specific assignments;
- `topics/topic_dictionary.csv`: terms, representative units, support and stability;
- `matrices/accident_topic_matrix.csv`: binary accident-level variables;
- `descriptive/topic_frequencies.csv`, co-occurrence and lift tables;
- `figures/pareto_validation_all_roles.png`: candidate configurations and Pareto frontiers;
- `parallel_runtime.json`: resolved worker count and Slurm CPU allocation;
- `theme_discovery_manifest.json`: run identity and selected configurations;
- `notebooks/recurrent_scenarios_results.ipynb`: post-run presentation notebook;
- `bayesian_networks/cv_log_likelihood.csv`;
- `bayesian_networks/arc_stability.csv`;
- `scenarios/scenario_catalog.csv`;
- `figures/`: descriptive plots and the two constrained networks;
- `audit_report.md`: resolved parameters, decisions, warnings and output index.
