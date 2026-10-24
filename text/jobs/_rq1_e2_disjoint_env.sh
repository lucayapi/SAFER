#!/usr/bin/env bash
set -euo pipefail
submit="${SLURM_SUBMIT_DIR:?Submit from SAFER or SAFER/text}"
if [[ -f "${submit}/text/jobs/_rq1_final_env.sh" ]]; then
  jobs="${submit}/text/jobs"
else
  jobs="${submit}/jobs"
fi
source "${jobs}/_rq1_final_env.sh" "${1:-run}"
export RQ1_E2D_SCRIPT="${SAT_LODO_TEXT_ROOT}/scripts/rq1_e2_disjoint.py"
export RQ1_E2D_ROOT="${RQ1_E2D_ROOT:-${SCRATCH:-$HOME}/safer_sat_rq1_e2_disjoint/runs/rq1_e2_disjoint_v1}"
[[ -f "${RQ1_E2D_SCRIPT}" ]] || { echo "Missing ${RQ1_E2D_SCRIPT}" >&2; exit 1; }
mkdir -p "${RQ1_E2D_ROOT}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
