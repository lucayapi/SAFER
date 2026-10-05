# Paired recovery experiment for the JRSS C manuscript

Run these commands from `text/` on the Slurm cluster. The experiment compares
the fixed reference factors with factors reconstructed from the **same 80% of
distinct accidents** in each replicate. The selected UMAP and HDBSCAN settings
are read from the archived carpentry and joinery run; the original grid search
is not repeated.

The prespecified statistical design is in `recurrent_scenarios/config.yaml`,
under `paired_recovery`: the number of replicates, sampling fraction, random
seed, matching thresholds and worker count. With `n_workers: auto`, the
program uses all CPUs allocated to its one Slurm job.

## Pilot (10 replicates)

```bash
N_REPLICATES=10 OUTPUT_DIR=recurrent_scenarios/runs/theme_discovery_audit/btp_carpentry_and_joinery/paired_recovery_pilot \
  sbatch jobs/run_recurrent_scenarios_paired_recovery.sh
```

The pilot checks the pipeline, but its recovery frequencies have large Monte
Carlo error and must not be used in the manuscript.

## Full experiment (200 replicates)

```bash
sbatch jobs/run_recurrent_scenarios_paired_recovery.sh
```

This is one Slurm job. Its 30 allocated CPUs run up to 30 independent
replications at once; when one completes, the worker takes the next remaining
replication. The job automatically writes the summary and figures only after
all 200 replicates succeed. A completed replicate is skipped on resubmission if
its design hash matches. The summary refuses to run until all 200 replicate
files are present and the source files still match their hashes.

The full output is under
`recurrent_scenarios/runs/theme_discovery_audit/btp_carpentry_and_joinery/paired_recovery/`:

- `design.json`: source-file hashes and analysis settings;
- `replicates/`: selected accident IDs, factor matches, and both learned graphs
  for every replicate;
- `factor_recovery_by_role.csv` and `factor_recovery_tau_*.csv`;
- `edge_recovery_tau_*.csv`: all 18 reference edges, joint endpoint and
  conditional edge recovery, paired differences and Monte Carlo standard errors;
- `scenario_recovery_tau_*.csv`: all 80 reference scenarios and their paired
  recovery estimates;
- `article_scenario_recovery.csv`: the manuscript's S1--S4 configurations,
  linked to their archived scenario IDs;
- `edge_recovery_real.pdf`, `scenario_recovery_real.pdf` and
  `observed_results.md`: vector figures and numerical interpretation generated
  from the actual outputs.

One-to-one factor matching maximises total unit-level Jaccard similarity within
each role. The main matching threshold is 0.50, with 0.40 and 0.60 sensitivity.
Scenario recovery requires all constituent factors to match and the observed
subsample support to reach the original rate, 5/417. It does not require the
pattern to remain closed in that subsample. A factor absent from the sampled
units is counted as unrecovered. The Bayesian network is re-estimated on **all**
reconstructed non-noise factors before reference edges are matched, so unmatched
factors can compete for parent sets.

The archive's earlier 500-resample fixed-factor bootstrap sampled accidents
with replacement. These new fixed-factor frequencies use 80% distinct
accidents and must be compared only with the paired reconstruction branch.
The output describes recovery under this perturbation scheme; it does not
estimate causal effects or posterior probabilities.
