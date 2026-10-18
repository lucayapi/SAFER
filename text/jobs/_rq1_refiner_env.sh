#!/usr/bin/env bash
# Reuse the pinned GOLD v3 SaT runtime; keep this campaign isolated.
set -euo pipefail
refiner_submit="${SLURM_SUBMIT_DIR:?Submit from SAFER or SAFER/text}"
if [[ -f "${refiner_submit}/text/scripts/rq1_final.py" ]]; then
  refiner_safer_root="${refiner_submit}"
else
  refiner_safer_root="$(cd "${refiner_submit}/.." && pwd)"
fi
if [[ -z "${SAT_LODO_BUNDLE:-}" ]]; then
  for refiner_candidate in \
    "${SCRATCH:-$HOME}/safer_sat_rq1_final/bundle/rq1_segmentation_final_v1" \
    "${refiner_safer_root}/annotation/rq1_final_v1/bundle" \
    "${refiner_safer_root}/annotation/lodo_sat_transfer/rq1_final_v1/bundle" \
    "${refiner_safer_root}/annotation/lodo_sat_transfer/rq1_final_v1"; do
    if [[ -f "${refiner_candidate}/manifest.json" &&
          -f "${refiner_candidate}/complete.json" ]]; then
      export SAT_LODO_BUNDLE="${refiner_candidate}"
      break
    fi
  done
fi
if [[ -f "${SLURM_SUBMIT_DIR:?}/text/jobs/_rq1_final_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/text/jobs/_rq1_final_env.sh" "${1:-run}"
else
  source "${SLURM_SUBMIT_DIR}/jobs/_rq1_final_env.sh" "${1:-run}"
fi

export RQ1_REFINER_M3_ROOT="${RQ1_REFINER_M3_ROOT:-${RQ1_FINAL_RUN_ROOT}}"
if [[ ! -f "${RQ1_REFINER_M3_ROOT}/target_btp/selection.json" &&
      -f "${safer_root}/rq1_segmentation_final_v1/target_btp/selection.json" ]]; then
  export RQ1_REFINER_M3_ROOT="${safer_root}/rq1_segmentation_final_v1"
fi
[[ -f "${RQ1_REFINER_M3_ROOT}/target_btp/selection.json" ]] || {
  echo "Six selected M3 rotations not found in ${RQ1_REFINER_M3_ROOT}." >&2
  exit 1
}
export RQ1_REFINER_OUT="${RQ1_REFINER_OUT:-${storage_root}/safer_sat_rq1_refiner/runs/rq1_refiner_exploratory_v1}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "${RQ1_REFINER_OUT}"
