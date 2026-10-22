# Generic components from the Qwen3-4B leaf run

From `text/` on the mesocentre:

```bash
sbatch jobs/run_curated_tdm_bn.sh
```

The job expects the completed leaf run at
`bayesian_network/runs/btp_building_construction_masonry_qwen3_4b_raw_leaf_umap_no_llm_20261022T004957Z`.
Set `LEAF_RUN=bayesian_network/runs/<name>` before `sbatch` if it has a
different directory name. The job uses 24 CPUs and requests 128 GB for 24 h.
It does not regenerate embeddings, clusters or LLM labels.

1. `bayesian_network/curate_leaf_components.py` reads every factual unit
   from the selected leaf assignments, including HDBSCAN noise. The explicit
   role-specific expressions in `component_definitions_leaf.yaml` assign
   zero or more components. Candidate components appearing in fewer than
   12 distinct accidents are omitted from the modelling matrix.
2. The exact TDM preflight tries the declared theme-support thresholds
   12, 16, 20, 25, 30, 40, 50, 60 and 80, in that order. It keeps the
   smallest threshold that passes the K=3 and K=1 resource bounds. Failed
   attempts, the selected threshold and the resolved configuration are saved
   under `theme_dictionary/runs/curated_leaf_selection/`. If none passes,
   the job stops without claiming an approximate result.
3. The Bayesian analysis fits the primary strict role-order network and a
   separate greedy within-role sensitivity network, then runs 100 conditional
   accident bootstraps.
4. The TDM fits K=3 and K=1 to the same curated matrix.

The curated data appear in
`bayesian_network/runs/curated_leaf_components_qwen3_4b/`.
Key files are `factor_dictionary.csv`,
`accident_factor_matrix.csv`, `unit_component_audit.csv`,
`review_samples.csv`, `component_cooccurrence.csv`, `audit.json`
and `curation_manifest.json`. The unit audit preserves each source cluster
and sentence. The bootstrap and network outputs appear in
`bayesian_network/runs/curated_leaf_components_qwen3_4b_network/`.
The TDM outputs appear in `theme_dictionary/runs/curated_leaf/`.
Slurm progress is printed to `slurm-curated_tdm_bn-<jobid>.out` and
errors to the matching `.err` file.

The component codebook is a reproducible, rule-assisted interpretation
of the corpus; it is not an independent expert adjudication. The
modelling outputs are descriptive and conditional on these definitions.
