#!/usr/bin/env bash
set -euo pipefail

submit="${SLURM_SUBMIT_DIR:?Submit from SAFER or SAFER/text}"
if [[ -f "${submit}/text/jobs/_rq1_e2_crossfit_env.sh" ]]; then
  jobs="${submit}/text/jobs"
elif [[ -f "${submit}/jobs/_rq1_e2_crossfit_env.sh" ]]; then
  jobs="${submit}/jobs"
else
  echo 'Submit from SAFER/ or SAFER/text/.' >&2
  exit 1
fi
export RQ1_E2X_ROOT="${RQ1_E2X_ROOT:-${SCRATCH:-$HOME}/safer_sat_rq1_refiner_crossfit/runs/rq1_e2_crossfit_exclusion_source_v1}"
source "${jobs}/_rq1_e2_crossfit_env.sh" "${1:-run}"
export RQ1_EXCLUSION_SCRIPT="${SAT_LODO_TEXT_ROOT}/scripts/rq1_e2_excluded.py"
export RQ1_EXCLUSION_ROOT="${RQ1_EXCLUSION_ROOT:-${SCRATCH:-$HOME}/safer_sat_rq1_exclusion_boundaries/runs/seed13_v2}"
[[ -f "${RQ1_EXCLUSION_SCRIPT}" ]] || {
  echo "Missing ${RQ1_EXCLUSION_SCRIPT}" >&2
  exit 1
}
mkdir -p "${RQ1_EXCLUSION_ROOT}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
