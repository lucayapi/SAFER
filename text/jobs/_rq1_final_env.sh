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
if [[ -f "${SLURM_SUBMIT_DIR}/text/scripts/rq1_final.py" ]]; then
  safer_root="${SLURM_SUBMIT_DIR}"
else
  safer_root="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
fi
export RQ1_FINAL_SOURCE="${storage_root}/safer_sat_indomain/bundle/sat12l_sm_gold_v3_indomain_v1"
if [[ ! -f "${RQ1_FINAL_SOURCE}/complete.json" &&
      -f "${safer_root}/safer_sat_indomain/bundle/sat12l_sm_gold_v3_indomain_v1/complete.json" ]]; then
  export RQ1_FINAL_SOURCE="${safer_root}/safer_sat_indomain/bundle/sat12l_sm_gold_v3_indomain_v1"
fi
export RQ1_FINAL_BUNDLE="${storage_root}/safer_sat_rq1_final/bundle/rq1_segmentation_final_v1"
if [[ ! -f "${RQ1_FINAL_BUNDLE}/complete.json" ]]; then
  for candidate in \
    "${safer_root}/annotation/rq1_final_v1/bundle" \
    "${safer_root}/annotation/lodo_sat_transfer/rq1_final_v1/bundle" \
    "${safer_root}/annotation/lodo_sat_transfer/rq1_final_v1"; do
    if [[ -f "${candidate}/complete.json" && -f "${candidate}/manifest.json" ]]; then
      export RQ1_FINAL_BUNDLE="${candidate}"
      break
    fi
  done
fi
export RQ1_FINAL_RUN_ROOT="${storage_root}/safer_sat_rq1_final/runs/rq1_segmentation_final_v1"
[[ -f "${RQ1_FINAL_BUNDLE}/complete.json" || -f "${RQ1_FINAL_SOURCE}/complete.json" ]] || {
  echo "Private RQ1 bundle missing. Expected manifest.json and complete.json in ${RQ1_FINAL_BUNDLE}/, ${safer_root}/annotation/lodo_sat_transfer/rq1_final_v1[/bundle]/, or source bundle ${RQ1_FINAL_SOURCE}/." >&2
  exit 1
}
if [[ -f "${RQ1_FINAL_BUNDLE}/complete.json" ]]; then
  export SAT_LODO_BUNDLE="${RQ1_FINAL_BUNDLE}"
else
  export SAT_LODO_BUNDLE="${RQ1_FINAL_SOURCE}"
fi
export SAT_LODO_RUN_ROOT="${RQ1_FINAL_RUN_ROOT}"
sat_gold_v3_setup "${1:-run}"

