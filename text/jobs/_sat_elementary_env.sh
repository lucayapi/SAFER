#!/usr/bin/env bash
# Reuse the pinned SaT-LoRA runtime and original RQ1 bundle, with separate outputs.

sat_elementary_setup() {
  local mode="${1:-run}"
  local storage_root="${SCRATCH:-$HOME}"
  export SAT_LODO_RUN_ROOT="${SAT_LODO_RUN_ROOT:-${storage_root}/safer_sat_lodo/runs/sat12l_sm_elementary_v1}"
  if [[ -f "${SLURM_SUBMIT_DIR}/text/jobs/_sat_lodo_gold_v3_env.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/text/jobs/_sat_lodo_gold_v3_env.sh"
  else
    source "${SLURM_SUBMIT_DIR}/jobs/_sat_lodo_gold_v3_env.sh"
  fi
  sat_gold_v3_setup "${mode}"
  local safer_root
  safer_root="$(cd "${SAT_LODO_TEXT_ROOT}/.." && pwd)"
  local -a candidates=(
    "${safer_root}/annotation/lodo_sat_transfer/sat12l_sm_elementary_v1"
    "${storage_root}/safer_private/sat12l_sm_elementary_v1"
    "${HOME}/safer_private/sat12l_sm_elementary_v1"
  )
  if [[ -n "${SAT_ELEMENTARY_PREPARED:-}" ]]; then
    candidates=("${SAT_ELEMENTARY_PREPARED}")
  fi
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}/manifest.json" && -f "${candidate}/complete.json" ]]; then
      export SAT_ELEMENTARY_PREPARED="${candidate}"
      echo "Paquet des unites elementaires : ${SAT_ELEMENTARY_PREPARED}"
      return 0
    fi
  done
  echo "Paquet sat12l_sm_elementary_v1 introuvable; copier le dossier prepare sur le cluster." >&2
  return 1
}
