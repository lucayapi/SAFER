# Reduced end-to-end Table 4 campaign

From the `SAFER/` repository root on the cluster, after copying `text/scripts/role_narrative_e2e_retrain.py`, the shared scoring module `text/scripts/role_narrative_e2e_baselines.py`, `text/jobs/role_narrative_e2e_retrain*.sh`, and `text/jobs/_role_narrative_e2e_retrain_env.sh`:

```bash
bash text/jobs/role_narrative_e2e_retrain_launch.sh
```

This submits four Slurm jobs: CPU preflight, GPU smoke, a GPU array of 54 tasks (`6 targets x 3 seeds x 3 full-fine-tuning methods`, at most three concurrent tasks), and CPU summary. There are 57 Slurm tasks in total. If the earlier CE/Triplet arrays must finish first, set `ROLE_E2E_AFTEROK=9172885:9172887` before the command; this adds a dependency to preflight and does not submit those jobs again.

The methods are CE, SupCon + LR and SoftTriple + LR, each trained for three epochs with the existing source-only selection protocol. SoftTriple predicts on M2 (source-calibrated SaT) and M3 (LoRA-adapted SaT) segments; CE and SupCon predict on M3 only. Target labels are used solely for final scoring. The previous classifier checkpoints were deleted, so these are new full training runs. No TF-IDF, frozen Qwen3, Triplet, PySBD or default SaT evaluation is launched here.

The default output is `$SCRATCH/safer_role_qwen3_narrative_e2e_reduced_v1/`, separate from the previous role campaign. Each task saves reference-unit classification metrics, per-segment role predictions, role-aware counts and F1, an audit marker and hashes. It verifies the segment predictions before deleting the heavy selected checkpoint. Completed verified tasks are skipped on restart. The selected epoch and regularisation come from the five source validation domains, never the target.

The summary produces `e2e_results/table4_display.csv` with five rows in Table 4 order and `e2e_results/table4_real_values.csv` with four unique configurations. M3 + SoftTriple appears twice in the display file with identical values. For each configuration, F1 is averaged over six target domains per seed, then reported as mean and sample standard deviation across the three seeds. `e2e_results/reference_unit_metrics.csv` retains the reference-unit scores for comparison with the previous campaign. The CSVs do not automatically rewrite the manuscript.

Preflight requires the unchanged `narrative_only_lodo_v1` bundle, M2 and M3 test predictions, the cleaned factual-unit CSVs, Qwen3 files, and `text/.venv` on the cluster. It validates signatures before GPU work. The GPU smoke runs all three adapted methods on small real subsets and checks that saved predictions remain valid after checkpoint cleanup.
