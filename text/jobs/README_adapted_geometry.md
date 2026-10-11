# Adapted-space error diagnostic

For the **reduced Qwen3/BTP pilot**, submit
`sbatch jobs/submit_adapted_geometry_pilot.sh` instead. It runs three adapted
methods x five seeds = 15 GPU refits and 15 CPU geometry tasks, each array with
at most four concurrent tasks. Its recipe is
`output/replication_recipes/adapted_geometry_qwen3_btp.yaml` and all outputs
are isolated under `output/adapted_geometry_pilot/`. Each refit still performs
the three-fold source CV and final source fit, then exports source and three
target embeddings. The final association compares methods on Chemistry--plastics
and Company. It does not test backbone or source-sector robustness.

The command below remains the **full 90-run** campaign.

From the cluster's `text/` directory, submit once:

```bash
sbatch jobs/submit_adapted_geometry.sh
```

The submit job creates `output/replication_recipes/adapted_geometry.yaml`, then
submits three dependent stages. The first stage refits 18 selected pipelines
(Qwen3, multilingual E5 and CamemBERT; BTP and Metallurgy; CE, SupCon and
SoftTriple) with five seeds each. It runs 90 GPU tasks, at most four
simultaneously. Every task requests one GPU and 30 CPUs, with 29 computation
threads. The second stage computes exact cosine source-neighbour indicators
and source-role centroid margins in the adapted representation from each refit.
It is a 90-task CPU array, also limited to four concurrent tasks. The last CPU
job combines the two common targets and fits accident-clustered binomial GEE
models with role, target and role-by-target adjustment.

The original campaigns deleted their checkpoints and adapted embeddings, so
these are **refits**, not reconstructions of the original fitted weights.
Embeddings and predictions are saved together under
`output/adapted_geometry/runs/`. Each refit's target BA is compared with its
historical counterpart in `output/adapted_geometry/analysis/historical_ba_comparison.csv`.
The geometry model always uses the refit's own predictions. A difference over
one BA point is flagged for review.

If every stage succeeds, final outputs appear under
`output/adapted_geometry/analysis/`: `adapted_geometry_diagnostics.pdf`,
`role_target_error_profiles.csv`, `seed_target_summary.csv`, `gee_coefficients.csv`,
`coefficient_spread_across_methods.csv`,
`historical_ba_comparison.csv` and `parameters.json`. Per-unit features stay
under `output/adapted_geometry/features/` and contain identifiers and labels,
but no raw narrative text. The PDF contains only aggregates.

The role margin uses the *true* target label and is diagnostic only. GEE gives
cluster-robust **associations**, not a causal or mixed-effects mechanism.
Comparison of coefficients across pipelines is exploratory and takes place
after standardizing each geometric predictor within its pipeline.
