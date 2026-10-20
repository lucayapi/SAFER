# TDM-CC-BU-K for accident-factor baskets

This is a standalone, descriptive implementation of the binary-union,
cross-category theme dictionary model in Yang and Deng (2023), *Generalized
Theme Dictionary Models for Association Pattern Discovery*, DOI
10.1214/22-AOAS1626. It reads the completed `Building construction and
masonry` accident-factor matrix; it does not recreate factors or call an LLM.

Run on the mesocentre from `text/`:

```bash
sbatch jobs/run_tdm_cc_bu.sh
```

That single job fits K=3 and then K=1. To submit two independent 24-CPU
array tasks in one command, use:

```bash
sbatch --array=1,3 jobs/run_tdm_cc_bu.sh
```

The tasks may run at the same time if the partition has sufficient
resources. Each writes to its own `k3/` or `k1/` directory under the same
hash-named run root, so checkpoints and reports cannot overwrite each other.
Each array task requests 24 CPUs and 32 GiB of RAM; two concurrent tasks
therefore reserve 64 GiB in total. This is a resource request, not a measured
peak. Check actual peak memory with Slurm accounting after the first run.
Each job writes its own comparison with the candidate screen. The fitted K=3
and K=1 summaries can be compared after both finish.

The default configuration uses an observed-support threshold of 12 for
multi-factor candidate themes. It still uses every accident, includes all
single-factor themes, and applies the same generative model. Its exact
preflight passed with 229 initial themes, 214,316 K=3 cover groups and 8,595
K=1 cover groups across the 1,899 distinct baskets. This threshold was chosen
after a complexity audit, so it must be reported with the results.

The originally proposed support-3 screen is retained separately. Its exact
preflight fails on the complete matrix: one 27-factor accident alone has more
than 10 million distinct cover groups. Run its diagnostic with:

```bash
CONFIG=theme_dictionary/config_support3.yaml PREFLIGHT=1 sbatch jobs/run_tdm_cc_bu.sh
```

The support-3 and support-12 candidate dictionaries are not interchangeable.

To verify that this limit reflects distinct cover partitions rather than
enumerated latent partitions, count the former without storing them:

```bash
python -m theme_dictionary.diagnose_cover_count \
  --config theme_dictionary/config_support3.yaml --cap 10000000
```

For an explicit exact-enumeration resource check without EM:

```bash
PREFLIGHT=1 sbatch jobs/run_tdm_cc_bu.sh
```

The default configuration includes all 2,382 accidents, including baskets
with no discovered factor. It mines all single factors and every cross-role
itemset of two to four factors observed at least twelve times. The initial
candidate catalog is the frequent-itemset comparator. K=3 is the primary
model; K=1 uses the same candidates and forbids overlap in cover partitions.
The K bound applies to maximal themes in a cover partition (equation 4.12),
not to optional proper subthemes in its latent group. Consequently a MAP
latent partition can contain overlapping proper subthemes even when K=1.
The roles are item categories, not causal restrictions.

The implementation sums cover-partition *groups*, integrating optional
proper subthemes as in equations (4.5)--(4.11) of the paper. EM has three
deterministic starts. At each pruning round, themes with deletion score below
`log(n)/2` are removed together and the model is refitted. The log score
follows the paper's K-restricted cover-partition likelihood; it is not a
held-out prediction score. A 1e-12 probability bound prevents floating-point
log(0) during EM, and convergence requires three successive relative score
changes below 1e-8. These numerical choices are implementation defaults,
not settings specified by the paper.

Results go to a hash-named subdirectory of
`theme_dictionary/runs/mesocentre_btp_building_construction_and_masonry/`.
For separate jobs, their files are in `k3/` and `k1/` below that directory:

- `input_audit.json`, `frequent_candidate_themes.csv`, resolved config and
  manifest with input/code hashes;
- `theme_catalog_k3.csv` and `theme_catalog_k1.csv`, plus B-theme views;
- `theme_counts_by_size_roles_k*.csv`, comparing candidates and retained
  themes by size and role combination;
- `accident_partitions_k*.csv`, `summary_k*.json`, and `method_comparison.csv`;
- `basket_fit_k*.csv` and `diagnostics_k*.json`, separating observed support,
  fitted basket probability, EM convergence, and theme elimination;
- `checkpoint_k*.json`, updated after every EM iteration for resubmission.

An exact fit can be combinatorially expensive for rich baskets. The configured
group limits are explicit resource guards: exceeding one stops the run with
a diagnostic, rather than silently truncating partitions or returning an
approximate estimate. Re-submit the same command to resume an interrupted EM
fit with identical files and configuration. After an input, config or code
change, use a fresh `RUN_DIR`.
