#!/usr/bin/env bash
# Check frozen GOLD, splits, grouped duplicate folds, and total private disk needs.
#SBATCH --job-name=rq1-e2x-prepare
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
python -u "${RQ1_E2X_SCRIPT}" prepare --bundle "${RQ1_FINAL_BUNDLE}" --out "${RQ1_E2X_ROOT}"
if [[ "${RQ1_E2X_REQUIRE_PILOT:-0}" == "1" ]]; then
  [[ -f "${RQ1_E2X_ROOT}/target_btp/refiner/complete.json" ]] || {
    echo 'The BTP pilot is not complete. Review it before submitting --remaining.' >&2
    exit 1
  }
  echo 'BTP pilot completion marker found; remaining rotations are authorized by this launch.'
fi
