#!/usr/bin/env bash
# Reuse the pinned SaT runtime and GOLD bundle. The filtered elementary
# training files are rebuilt by the smoke job, so no old elementary bundle
# needs to be copied to the cluster.

sat_elementary_no_nicollin_setup() {
  local mode="${1:-run}"
  local storage_root="${SCRATCH:-$HOME}"
  export SAT_NO_NICOLLIN_RUN_ROOT="${SAT_NO_NICOLLIN_RUN_ROOT:-${storage_root}/safer_sat_lodo/runs/sat12l_sm_elementary_no_nicollin_v1}"
  export SAT_LODO_RUN_ROOT="${SAT_NO_NICOLLIN_RUN_ROOT}"
  source "${SLURM_SUBMIT_DIR}/text/jobs/_sat_lodo_gold_v3_env.sh"
  sat_gold_v3_setup "${mode}"
}
