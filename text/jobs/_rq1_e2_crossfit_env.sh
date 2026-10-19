#!/usr/bin/env bash
# Isolated runtime and paths for the cross-fitted E2 campaign.
set -euo pipefail
crossfit_submit="${SLURM_SUBMIT_DIR:?Submit from SAFER or SAFER/text}"
if [[ -f "${crossfit_submit}/text/jobs/_rq1_final_env.sh" ]]; then
  crossfit_jobs="${crossfit_submit}/text/jobs"
elif [[ -f "${crossfit_submit}/jobs/_rq1_final_env.sh" ]]; then
  crossfit_jobs="${crossfit_submit}/jobs"
else
  echo 'Submit from SAFER/ or SAFER/text/.' >&2
  exit 1
fi
source "${crossfit_jobs}/_rq1_final_env.sh" "${1:-run}"
crossfit_storage="${SCRATCH:-$HOME}"
export RQ1_E2X_ROOT="${RQ1_E2X_ROOT:-${crossfit_storage}/safer_sat_rq1_refiner_crossfit/runs/rq1_e2_crossfit_v2}"
export RQ1_E2X_SCRIPT="${SAT_LODO_TEXT_ROOT}/scripts/rq1_e2_crossfit.py"
[[ -f "${RQ1_E2X_SCRIPT}" ]] || { echo "Missing ${RQ1_E2X_SCRIPT}" >&2; exit 1; }
mkdir -p "${RQ1_E2X_ROOT}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
