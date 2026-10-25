#!/usr/bin/env bash
# Same pinned runtime and immutable LODO bundles, isolated full-FT outputs.

sat_elementary_full3_setup() {
  local mode="${1:-run}"
  local storage_root="${SCRATCH:-$HOME}"
  export SAT_ELEMENTARY_FULL3_RUN_ROOT="${SAT_ELEMENTARY_FULL3_RUN_ROOT:-${storage_root}/safer_sat_lodo/runs/sat12l_sm_elementary_full3_v1}"
  export SAT_LODO_RUN_ROOT="${SAT_ELEMENTARY_FULL3_RUN_ROOT}"
  source "${SLURM_SUBMIT_DIR}/text/jobs/_sat_elementary_env.sh"
  sat_elementary_setup "${mode}"
}
