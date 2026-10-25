#!/usr/bin/env bash
# Isolate the checkpoint-selected rerun from the earlier final-only results.

sat_elementary_checkpoint_setup() {
  local mode="${1:-run}"
  local storage_root="${SCRATCH:-$HOME}"
  export SAT_ELEMENTARY_CHECKPOINT_RUN_ROOT="${SAT_ELEMENTARY_CHECKPOINT_RUN_ROOT:-${storage_root}/safer_sat_lodo/runs/sat12l_sm_elementary_checkpoint_v2}"
  export SAT_LODO_RUN_ROOT="${SAT_ELEMENTARY_CHECKPOINT_RUN_ROOT}"
  if [[ -f "${SLURM_SUBMIT_DIR}/text/jobs/_sat_elementary_env.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/text/jobs/_sat_elementary_env.sh"
  else
    source "${SLURM_SUBMIT_DIR}/jobs/_sat_elementary_env.sh"
  fi
  sat_elementary_setup "${mode}"
}
