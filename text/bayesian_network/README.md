# Descriptive Bayesian-network and recurrent-scenario analysis

This directory contains a configurable, single-job analysis for accident
corpora and existing subfamilies. It keeps valid factual units labelled A0,
A1, B or C, joins units to embeddings through the configured identifiers, and
excludes `AUTRE` before factor discovery. Unmatched eligible units cause an
explicit error; the job does not generate embeddings or switch models.

## Submit one Slurm job

From the `text/` directory on the cluster, the default job uses the general
BTP configuration:

```bash
sbatch jobs/run_bayesian_network.sh
```

To analyse the Building construction and masonry family with the refreshed
Qwen embeddings and HDBSCAN EOM selection:

```bash
CONFIG=bayesian_network/config_btp_building_construction_and_masonry.yaml REESTIMATE=1 sbatch jobs/run_bayesian_network.sh
```

The OpenAI key must be available in `text/.env` or in the job environment.
Each run receives a separate timestamped directory and annotation cache.
`REESTIMATE=1` forces the UMAP/HDBSCAN candidate and resampling computations
to run again; otherwise compatible completed discovery results and LLM labels
are reused.

## Analysis

1. Audit the role filtering and one-to-one unit/embedding join.
2. Evaluate the configured UMAP/HDBSCAN candidates on all eligible units.
   Use the existing Pareto/Tchebycheff selection and grouped accident
   resampling to choose a stable role-specific partition. These resamples
   assess clustering selection and stability; no accident sample is reserved.
3. Annotate the selected factors with the LLM and build one binary
   accident-by-factor matrix covering the complete eligible corpus.
4. Report factor prevalence and role distribution, the number of factors per
   accident, and factor co-occurrences.
5. Learn the primary role-ordered network by exact decomposable BIC search,
   plus a greedy sensitivity network that allows within-role edges. CPTs use
   Jeffreys Beta(0.5, 0.5) smoothing.
6. Extract graph-connected scenario motifs that occur in at least five
   accidents in the complete matrix. Report their observed support, lift over
   independence and observed conditional consequence frequencies, alongside
   Monte Carlo network probabilities. These are descriptive summaries; no
   significance tests or held-out predictive scores are computed.
7. Run 100 accident bootstraps to summarize edge and scenario selection
   stability. Bootstrap results are conditional on the full-corpus factor
   definitions, LLM labels and unit-to-factor assignments.

Network edges describe conditional statistical dependence; they are not
causal effects. Scenario support describes the configured corpus and factor
assignments. Bootstrap stability does not include uncertainty from factor
discovery, LLM annotation or unit-to-factor assignment.

## Main outputs

Each run directory contains the resolved configuration and input/code hashes,
alignment and exclusion audits, factor dictionary and unit assignments,
`accident_factor_matrix.csv`, factor prevalence and co-occurrence summaries,
both network structures and CPTs, descriptive scenario results, candidate
catalog coverage/redundancy summaries, and bootstrap edge/scenario stability.
No accident sample is reserved for out-of-sample evaluation, and no predictive
performance table is produced.

The UMAP/HDBSCAN parameter grid, internal resampling settings, and compute
limits are configured in the selected YAML file. Bootstrap and candidate
evaluation use the Slurm CPU allocation subject to `parallel.max_workers`.
