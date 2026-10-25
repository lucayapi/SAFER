#!/usr/bin/env bash
# Runtime and shared data folds for the narrative-only M0-M3 comparison.

sat_narrative_only_setup() {
  local mode="${1:-run}"
  local storage_root="${SCRATCH:-$HOME}"
  if [[ -f "${SLURM_SUBMIT_DIR}/text/scripts/sat_narrative_only_lodo.py" ]]; then
    SAT_LODO_TEXT_ROOT="${SLURM_SUBMIT_DIR}/text"
  elif [[ -f "${SLURM_SUBMIT_DIR}/scripts/sat_narrative_only_lodo.py" ]]; then
    SAT_LODO_TEXT_ROOT="${SLURM_SUBMIT_DIR}"
  else
    echo "Submit from SAFER/ or SAFER/text/." >&2
    return 1
  fi
  cd "${SAT_LODO_TEXT_ROOT}"
  source jobs/_env.sh
  source jobs/_sat_lodo_runtime.sh
  load_hpc_modules
  export SAT_LODO_RUNTIME_ROOT="${storage_root}/safer_sat_lodo/runtime_gold_v3"
  export SAT_LODO_VENV="${SAT_LODO_RUNTIME_ROOT}/.venv1"
  export SAT_LODO_REQUIREMENTS="${SAT_LODO_TEXT_ROOT}/annotation/requirements_sat_lodo.txt"
  export SAT_WTPSPLIT_REPO="${SAT_WTPSPLIT_REPO:-${SAT_LODO_TEXT_ROOT}/vendor/wtpsplit}"
  export SAT_NARRATIVE_RUN_ROOT="${SAT_NARRATIVE_RUN_ROOT:-${storage_root}/safer_sat_lodo/runs/narrative_only_lodo_v1}"
  setup_hf_cache_env
  if [[ "${mode}" == "smoke" ]]; then
    sat_lodo_bootstrap
  else
    sat_lodo_check_runtime
  fi
  source "${SAT_LODO_VENV}/bin/activate"
  export PYTHONPATH="${SAT_LODO_TEXT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true PYTHONUNBUFFERED=1
  mkdir -p "${SAT_NARRATIVE_RUN_ROOT}"

  local safer_root candidate
  safer_root="$(cd "${SAT_LODO_TEXT_ROOT}/.." && pwd)"
  local -a candidates=(
    "${safer_root}/annotation/lodo_sat_transfer/narrative_only_lodo_v1"
    "${storage_root}/safer_private/narrative_only_lodo_v1"
    "${HOME}/safer_private/narrative_only_lodo_v1"
  )
  if [[ -n "${SAT_NARRATIVE_PREPARED:-}" ]]; then
    candidates=("${SAT_NARRATIVE_PREPARED}")
  fi
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}/manifest.json" && -f "${candidate}/complete.json" ]]; then
      export SAT_NARRATIVE_PREPARED="${candidate}"
      echo "Shared narrative-only folds: ${SAT_NARRATIVE_PREPARED}"
      return 0
    fi
  done
  echo "Paquet narrative_only_lodo_v1 introuvable; copier le dossier préparé sur le cluster." >&2
  return 1
}
