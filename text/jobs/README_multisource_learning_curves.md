# Equal-size multi-source learning curves

From the `text/` directory on the cluster:

```bash
sbatch jobs/submit_multisource_learning_curves.sh
```

The launcher prepares five nested whole-accident draws for total source-unit
budgets 2,000, 4,000, 6,000, 9,000, and 12,000. It then submits **495 GPU
fits**: seven source sets at the first three budgets, six at 9,000 and 12,000,
times five draws and three training seeds. Chemistry–plastics alone is omitted
above 6,000 because only 6,157 labelled units are available. Every feasible
mixture has approximately the same **total** unit count as a single-source
condition at that budget. Source quotas differ by at most one unit, and actual
counts can differ slightly because complete accidents are selected.
At 12,000 units, BTP + Chemistry uses nearly all Chemistry units; five
Chemistry draws at that quota will overlap substantially and should not be
treated as independent new corpora.

The GPU array permits at most four tasks at once, subject to scheduler
availability. Each fit requests one GPU and 30 CPUs and uses 29 compute
threads. The launcher and analysis also request 30 CPUs. Preparation and
analysis are linked to the array by SLURM `afterok` dependencies; analysis
will not start if any fit fails. Completed fit tasks can be rerun safely: the
pipeline verifies their saved predictions and skips training.

All fits reuse the fixed Qwen3 full-encoder cross-entropy pipeline from the
earlier multi-source campaign: class weights, no projector, three epochs,
batch size 32, and unchanged optimizer settings. Company is evaluation-only;
no target score selects a source set, budget, or model. The full design and
source counts are saved in `design.json` and `source_samples.csv`.

Results are written to `output/multisource_learning_curves/run_<launcher-job-id>/`.
The `analysis/` directory contains per-run metrics, mean curves with run and
draw dispersion, paired differences against BTP, 2,000 paired accident-level
bootstrap intervals, and two PDF figures. The mean-of-constituent-singletons
contrast is available only where all constituent single-source conditions
are feasible. Bootstrap intervals cover target-accident sampling conditional
on the fitted models; seed and source-draw variation are reported separately.

The Company corpus was already examined in the earlier campaign. This is an
exploratory follow-up on that same target, not an untouched confirmation of a
general rule. The campaign tests whether the observed benefit of source
mixing changes with sample size; it does not prove a semantic-coverage cause.
