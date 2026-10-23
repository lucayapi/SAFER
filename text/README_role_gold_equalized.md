# Volume-matched single-source versus multisource classification

Run from the `SAFER/` root on the cluster:

```bash
sbatch text/jobs/role_gold_equalized.sh
```

All jobs in this campaign use `text/.venv2` by default. The same venv is the
default for other jobs that use the shared `setup_text_job_env` helper. Set
`SAFER_TEXT_VENV` (or `ROLE_GOLD_VENV` for this campaign) only if the cluster
stores that venv at a different path.

The launcher prepares and runs both arms on the same narrative-v3 GOLD and the
same frozen RQ1 test narratives. It limits each of the two arms to one GPU job
at a time, so at most two GPU jobs run concurrently. It uses the existing
five-role E5-large experiments (frozen E5 + logistic regression, CE, SupCon,
SoftTriple) and the existing three training seeds.

Each model trains on exactly 2,400 units: 480 for each role (`A0`, `A1`, `B`,
`C`, `AUTRE`). A single-source model draws all 480 examples per role from one
allowed source corpus. A multisource LODO model draws 96 examples per role from
each of its five source corpora. Validation is also matched at 375 units and
75 per role: the multisource arm takes 15 per role per source domain. Sampling
uses a fixed seed and selected `doc_id`s are saved and fingerprinted before
training. Sampling occurs only inside the frozen training and validation
partitions; the target test narratives are unchanged.

The full-volume campaigns remain untouched. Results are written under
`$SCRATCH/safer_role_gold_v3_equalized/` (or the repository root if `SCRATCH`
is unset), in `single_source_equalized/`, `lodo_equalized/`, and `results/`.
The final comparison includes per-target and macro Balanced Accuracy and
Macro-F1, with paired 95% cluster-bootstrap confidence intervals that
resample test narratives (`accident_id`) rather than individual fact units.
The same narrative resample is applied to the single-source and LODO
predictions for each target. The intervals are conditional on the fitted
checkpoints and fixed training seeds; they do not include training-seed or
domain-population uncertainty. Three-seed variability remains available in
the arm summaries.

The comparison controls total training size and class composition;
the source-domain composition is the experimental difference. It therefore
estimates the observed effect of adding source domains for these corpora and
this sampling protocol, not a universal causal effect of source count.

The comparison job computes the bootstrap automatically after both campaign
arms finish. It writes `results/equalized_bootstrap_by_target.csv`, the
per-target and macro comparison CSVs, and
`results/equalized_bootstrap_manifest.json`. To recompute locally from completed
predictions, run:

```bash
python text/scripts/role_gold_compare_equalized.py --run-root /path/to/safer_role_gold_v3_equalized
```
