# Multi-source EPICEA → Company pilot

From the `text/` directory, submit once:

```bash
sbatch jobs/submit_multisource_dg.sh
```

The launcher first prepares source samples, then queues one array of **126 GPU
fits** (`7 source sets × [5 controlled draws + 1 all-data draw] × 3 training
seeds`). At most four array tasks can run simultaneously. Every fit requests
one GPU and 30 CPUs and uses 29 compute threads. An analysis job starts only
after all fits succeed. SLURM scheduling may start fewer than four tasks when
resources are unavailable.

The seven source sets are BTP, metallurgy, chemistry–plastics, each pair, and
all three together. The controlled arm uses approximately 6,000 units per
condition: 6,000 from a single source, 3,000 from each source in a pair, or
2,000 from each of three sources. Entire accidents are sampled; nested source
subsamples and five independent draws permit checking sampling variation.
The all-data arm uses every annotated unit in each source. The exact unit,
accident, and role counts are saved in `source_samples.csv`. The Company corpus
is never included in a source mixture.

All fits use the same selected Qwen3 CE architecture: full encoder, no
projector, class-balanced CE, three epochs, batch size 32, source-only fixed
hyperparameters. **No target-based tuning or per-mixture hyperparameter
search occurs.** The fixed recipe makes the mixture comparison directly
interpretable as an initial diagnostic; it does not establish that each
source set has reached its own optimum. If this pilot reveals a meaningful
gain, a separate source-only validation study can test whether it survives
per-mixture tuning.

Output goes to `output/multisource_dg/run_<launcher-job-id>/`. The directory
contains the immutable campaign configuration, sampled source CSVs, task
manifest, source sample counts, target predictions and metrics for each run,
SLURM logs, and `analysis/` tables. The primary readout is Company balanced
accuracy for each multi-source condition versus the mean of its constituent
single-source conditions, compared within the same draw and seed. The
analysis also gives direct BTP contrasts, macro-F1, accuracy, and per-role
precision and recall. The 2,000 paired bootstrap resamples use complete
Company accidents and describe target-sample uncertainty conditional on the
fitted models. Run and draw standard deviations are reported separately.

The analysis fails if any fit is incomplete or if target keys or gold labels
do not match across runs. The job keeps prediction identifiers and labels but
removes duplicate raw-narrative and embedding exports from completed fits.
