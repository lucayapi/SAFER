#!/usr/bin/env bash
# Common paths for the isolated RQ1 campaign. Source from a Slurm job.
set -euo pipefail

if [[ -f "${SLURM_SUBMIT_DIR}/text/jobs/_sat_lodo_gold_v3_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/text/jobs/_sat_lodo_gold_v3_env.sh"
elif [[ -f "${SLURM_SUBMIT_DIR}/jobs/_sat_lodo_gold_v3_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_sat_lodo_gold_v3_env.sh"
else
  echo 'Submit from SAFER/ or SAFER/text/.' >&2
  exit 1
fi

storage_root="${SCRATCH:-$HOME}"
export RQ1_FINAL_SOURCE="${storage_root}/safer_sat_indomain/bundle/sat12l_sm_gold_v3_indomain_v1"
export RQ1_FINAL_BUNDLE="${storage_root}/safer_sat_rq1_final/bundle/rq1_segmentation_final_v1"
export RQ1_FINAL_RUN_ROOT="${storage_root}/safer_sat_rq1_final/runs/rq1_segmentation_final_v1"
[[ -f "${RQ1_FINAL_SOURCE}/complete.json" ]] || {
  echo "Private source bundle missing: ${RQ1_FINAL_SOURCE}" >&2
  exit 1
}
export SAT_LODO_BUNDLE="${RQ1_FINAL_SOURCE}"
export SAT_LODO_RUN_ROOT="${RQ1_FINAL_RUN_ROOT}"
sat_gold_v3_setup "${1:-run}"

