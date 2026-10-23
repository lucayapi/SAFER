#!/usr/bin/env bash
# Final paired summary for the volume-matched single-source and LODO runs.
# Submitted automatically by role_gold_equalized.sh after both arms finish.
#SBATCH --job-name=role-gold-eq-summary
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/text/jobs/_role_gold_env.sh"
python -u scripts/role_gold_compare_equalized.py
