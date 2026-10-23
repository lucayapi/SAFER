#!/usr/bin/env bash
# Launch the matched-volume single-source and five-source comparison.
# Run from SAFER/: sbatch text/jobs/role_gold_equalized.sh
#SBATCH --job-name=role-gold-equal-launch
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
if [[ -d "${SLURM_SUBMIT_DIR:?}/text/jobs" ]]; then
  bash "${SLURM_SUBMIT_DIR}/text/jobs/_role_gold_equalized_launch.sh"
else
  bash "${SLURM_SUBMIT_DIR}/jobs/_role_gold_equalized_launch.sh"
fi
