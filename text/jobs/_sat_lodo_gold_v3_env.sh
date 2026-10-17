#!/usr/bin/env bash
# Shared private paths and pinned LoRA runtime for the narrative_v3 RQ1 experiment.

sat_gold_v3_setup() {
  local mode="${1:-run}"
  if [[ -f "${SLURM_SUBMIT_DIR}/text/scripts/sat_lodo_gold_v3.py" ]]; then
    SAT_LODO_TEXT_ROOT="${SLURM_SUBMIT_DIR}/text"
  elif [[ -f "${SLURM_SUBMIT_DIR}/scripts/sat_lodo_gold_v3.py" ]]; then
    SAT_LODO_TEXT_ROOT="${SLURM_SUBMIT_DIR}"
  else
    echo "Submit from SAFER/ or SAFER/text/." >&2
    return 1
  fi
  cd "${SAT_LODO_TEXT_ROOT}"
  source jobs/_env.sh
  source jobs/_sat_lodo_runtime.sh
  load_hpc_modules
  local storage_root="${SCRATCH:-$HOME}"
  # Dedicated environment for this experiment: never reuse the cluster's old venv.
  export SAT_LODO_RUNTIME_ROOT="${storage_root}/safer_sat_lodo/runtime_gold_v3"
  export SAT_LODO_VENV="${SAT_LODO_RUNTIME_ROOT}/.venv1"
  export SAT_LODO_BUNDLE="${SAT_LODO_BUNDLE:-${storage_root}/safer_private/sat12l_sm_gold_v3_rq1}"
  export SAT_LODO_RUN_ROOT="${SAT_LODO_RUN_ROOT:-${storage_root}/safer_sat_lodo/runs/sat12l_sm_gold_v3_rq1}"
  sat_lodo_set_paths
  setup_hf_cache_env
  if [[ "${mode}" == "smoke" ]]; then
    sat_lodo_bootstrap
  else
    sat_lodo_check_runtime
  fi
  source "${SAT_LODO_VENV}/bin/activate"
  export PYTHONPATH="${SAT_LODO_TEXT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  export TOKENIZERS_PARALLELISM=false
  export WANDB_DISABLED=true
  export PYTHONUNBUFFERED=1
  mkdir -p "${SAT_LODO_RUN_ROOT}"
}
