# BTP Bayesian network analysis

This directory contains a self-contained orchestration layer for the full BTP
corpus. It keeps only valid factual units labelled A0, A1, B or C. The unit
table is joined to the E5 export by the globally unique `doc_id` (the source
`fact_id` is only an ordinal within each accident); unmatched valid units
cause an explicit error by default. There is no encoder fallback.

## Submit one Slurm job

From the `text/` directory on the cluster:

```bash
sbatch jobs/run_bayesian_network_btp.sh
```

Add `REESTIMATE=1` only when intentionally forcing clustering and stability
fits to run again. LLM labels are fingerprinted and cached in the run directory,
so a restarted job reuses completed labels when given the same run directory.
To resume, submit with `RUN_DIR=bayesian_network/runs/<existing-run-name>`.
The OpenAI key must be available in `text/.env` or the job environment.

The default E5 input is `text/embeddings/gold_v3/e5/btp.csv`. If it is absent,
the job stops before discovery and says exactly which file is required. It does
not switch to Qwen embeddings or generate new embeddings.

## Pipeline

1. Audit role filtering and the ID-based embedding join.
2. Split accidents 70/30 (seed 42); all UMAP/HDBSCAN discovery and LLM naming
   use training units only.
3. Evaluate the configured UMAP/HDBSCAN grid and resampling stability, then
   select each role's partition with the existing Pareto/Tchebycheff procedure.
4. Transform validation embeddings with the fitted UMAP and assign them using
   HDBSCAN approximate prediction.
5. Build accident-by-factor matrices and their frequency, role, count and
   co-occurrence summaries.
6. Learn the primary exact role-constrained BIC network and a greedy
   within-role sensitivity network. CPTs use Jeffreys Beta(0.5, 0.5) smoothing.
7. Freeze training motifs (one or two A factors, one B and optional C; support
   at least five), evaluate them on validation, and run 100 independently seeded
   accident bootstraps in parallel. Each completed replicate is checkpointed
   so an interrupted job can resume.

Network edges represent conditional statistical dependence; they are not
causal effects. Validation metrics are conditional on the factor definitions
and memberships learned from training data. Wilson intervals treat accidents
as independent. Bootstrap computation likewise holds the discovered factors
fixed and repeats network/motif selection.

## Main outputs

Each execution creates `runs/btp_<UTC timestamp>/` with resolved config,
input hashes, exclusion/alignment/split audits, train and validation
assignments and matrices, factor dictionary, BN structures and CPTs,
validation metrics and scenario tests, and bootstrap stability tables.

The grid and computational settings are in `config.yaml`. The default grid has
16 UMAP/HDBSCAN candidates per role and 10 accident resamples per candidate;
these can be expanded before a production run.
Bootstrap parallelism uses the same Slurm-aware worker limit as discovery and
can be disabled with `network.bootstrap_parallel: false`.
