# E5 embeddings for existing macro-activity families

The existing BTP, CAOU and metallurgie family memberships are shared with the
Qwen exports. They are not recomputed or relabelled for E5.

From the `text/` directory, create the family-level E5 tables with:

```bash
python annotation/export_family_embeddings.py
```

The exporter matches each family's unit `fact_id` to the source E5 `doc_id`,
checks for duplicate or missing IDs, and writes `multilingual_e5_large_*.csv`
files under `embeddings/families/{btp,caou,metallurgie}/`. It also writes an E5
family manifest in each corpus directory. The source E5 tables must be present
under `embeddings/multilingual_e5_large/`.

To run theme discovery on one of these family datasets, use the E5 config and
the same family ID used for Qwen:

```bash
python recurrent_scenarios/run_theme_discovery.py \
  --config recurrent_scenarios/config_e5.yaml \
  --dataset btp_electrical_installation
```

On Slurm, from `text/`:

```bash
CONFIG_PATH=recurrent_scenarios/config_e5.yaml \
DATASET=btp_electrical_installation REESTIMATE=1 \
sbatch jobs/run_recurrent_scenarios_theme_discovery.sh
```

E5 discovery outputs are kept under `recurrent_scenarios/runs/theme_discovery_e5/`
so they do not overwrite Qwen runs.
