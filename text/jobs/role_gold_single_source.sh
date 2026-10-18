#!/usr/bin/env bash
# From SAFER/: sbatch text/jobs/role_gold_single_source.sh
#SBATCH --job-name=role-gold-single-launch
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
if [[ -d "${SLURM_SUBMIT_DIR:?}/text/jobs" ]]; then
  bash "${SLURM_SUBMIT_DIR}/text/jobs/_role_gold_launch.sh" single_source
else
  bash "${SLURM_SUBMIT_DIR}/jobs/_role_gold_launch.sh" single_source
fi
