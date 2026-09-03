# Legacy latent-family analysis

The latent-family Bayesian network pipeline (Structural EM, selection of K,
posterior families, constrained MPE) is **retained for reproducibility of
exploratory analyses** but is **not used for the primary recurrent-scenario
results**.

## Primary pipeline (default)

- `analysis_mode: global_bn_scenario_mining` in `config.yaml`
- Notebook: `notebooks/recurrent_scenarios_bn_analysis_{dataset}.ipynb`
- Outputs:
  - `runs/.../accident_factor_matrix/` — frozen matrix + dictionary + audit
  - `runs/.../global_bayesian_network/` — global BN fit + bootstrap
  - `runs/.../recurrent_scenarios/` — empirical scenario mining
  - `runs/.../figs_ch4/` — manuscript figures

## Legacy mode

To rerun the exploratory latent-family model:

```python
config["analysis_mode"] = "legacy_latent_family"
# Use scenario_pipeline.run_frozen_bn_analysis(..., output_dir=run_dir / "legacy_latent_family")
```

Legacy code lives in:

- `scenario_pipeline.py` — `fit_latent_bn_analysis`, `extract_latent_bn_scenarios`
- `bn_reporting.py` — K selection, family weights, latent diagnostics

Do not mix legacy outputs with primary `recurrent_scenarios/` results.
