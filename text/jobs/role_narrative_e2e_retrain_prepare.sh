#!/usr/bin/env bash
#SBATCH --job-name=role-e2e-reduced-prepare
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/text/jobs/_role_narrative_e2e_retrain_env.sh"
python -u scripts/role_narrative_e2e_retrain.py prepare
