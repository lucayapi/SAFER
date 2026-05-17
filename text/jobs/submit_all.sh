#!/bin/bash
# Soumission séquentielle (adapter partition/gpu selon le Mésocentre).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Logs SLURM : slurm-<job_name>-<job_id>.out|.err dans ce dossier (jobs/).

sbatch export_raw_embeddings.sh
sbatch train_scgm_text.sh
sbatch train_batch_triplet.sh
sbatch train_softtriple.sh
sbatch train_supcon.sh
echo "Jobs soumis. Suivi : squeue -u \$USER"
echo "Après complétion : sbatch compare_methods.sh"
