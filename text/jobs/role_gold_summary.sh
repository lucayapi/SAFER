#!/usr/bin/env bash
#SBATCH --job-name=role-gold-summary
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/text/jobs/_role_gold_env.sh"
python -u scripts/role_gold_campaign.py summary --experiment "${ROLE_GOLD_EXPERIMENT:?}"
