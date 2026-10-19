#!/usr/bin/env bash
# Verify complete target outputs and write tables, paired bootstrap, and chart.
#SBATCH --job-name=rq1-e2x-summary
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
if [[ -f "${SLURM_SUBMIT_DIR:?}/text/jobs/_rq1_e2_crossfit_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/text/jobs/_rq1_e2_crossfit_env.sh"
else
  source "${SLURM_SUBMIT_DIR}/jobs/_rq1_e2_crossfit_env.sh"
fi
scope="${RQ1_E2X_SCOPE:?Set RQ1_E2X_SCOPE=pilot or all}"
python -u "${RQ1_E2X_SCRIPT}" summary --bundle "${RQ1_FINAL_BUNDLE}" \
  --out "${RQ1_E2X_ROOT}" --scope "${scope}"
