#!/usr/bin/env bash
# Separate cleaned-narrative run; reuse the pinned narrative_v3 SaT-LoRA runtime.

sat_narrative_only_setup() {
  local mode="${1:-run}"
  local storage_root="${SCRATCH:-$HOME}"
  if [[ -f "${SLURM_SUBMIT_DIR}/text/jobs/_sat_lodo_gold_v3_env.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/text/jobs/_sat_lodo_gold_v3_env.sh"
  else
    source "${SLURM_SUBMIT_DIR}/jobs/_sat_lodo_gold_v3_env.sh"
  fi
  sat_gold_v3_setup "${mode}"
  local safer_root candidate
  safer_root="$(cd "${SAT_LODO_TEXT_ROOT}/.." && pwd)"
  local -a candidates=(
    "${safer_root}/annotation/lodo_sat_transfer/sat12l_sm_narrative_only_v1"
    "${storage_root}/safer_private/sat12l_sm_narrative_only_v1"
    "${HOME}/safer_private/sat12l_sm_narrative_only_v1"
  )
  if [[ -n "${SAT_NARRATIVE_PREPARED:-}" ]]; then
    candidates=("${SAT_NARRATIVE_PREPARED}")
  fi
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}/manifest.json" && -f "${candidate}/complete.json" ]]; then
      export SAT_NARRATIVE_PREPARED="${candidate}"
      echo "Paquet SaT récits nettoyés : ${SAT_NARRATIVE_PREPARED}"
      export SAT_NARRATIVE_RUN_ROOT="${SAT_NARRATIVE_RUN_ROOT:-${storage_root}/safer_sat_lodo/runs/sat12l_sm_narrative_only_v1}"
      mkdir -p "${SAT_NARRATIVE_RUN_ROOT}"
      return 0
    fi
  done
  echo "Paquet sat12l_sm_narrative_only_v1 introuvable; copier le dossier préparé sur le cluster." >&2
  return 1
}
