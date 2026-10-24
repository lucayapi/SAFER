#!/usr/bin/env bash
# Shared, pinned runtime and private paths for the equal-volume SaT experiment.
set -euo pipefail

if [[ -f "${SLURM_SUBMIT_DIR:?}/text/scripts/rq1_seg_equal_volume.py" ]]; then
  safer_root="${SLURM_SUBMIT_DIR}"
else
  safer_root="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
fi
text_root="${safer_root}/text"
cd "${text_root}"
source jobs/_env.sh
source jobs/_sat_lodo_runtime.sh
load_hpc_modules
storage_root="${SCRATCH:-$HOME}"

find_private() {
  local suffix="$1" candidate
  for candidate in "${storage_root}/${suffix}" "${HOME}/${suffix}" "${safer_root}/${suffix}"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

if [[ -z "${RQ1_EQUAL_SOURCE_BUNDLE:-}" ]]; then
  RQ1_EQUAL_SOURCE_BUNDLE="$(find_private 'safer_sat_indomain/bundle/sat12l_sm_gold_v3_indomain_v1')" || {
    echo 'Private single-source GOLD bundle not found.' >&2
    exit 1
  }
fi
export RQ1_EQUAL_SOURCE_BUNDLE
export RQ1_EQUAL_REFERENCE="${RQ1_EQUAL_REFERENCE:-${RQ1_EQUAL_SOURCE_BUNDLE}/equal_volume_reference.json}"
export RQ1_EQUAL_RUN_ROOT="${RQ1_EQUAL_RUN_ROOT:-${storage_root}/safer_sat_equal_volume_v2}"
[[ -f "${RQ1_EQUAL_SOURCE_BUNDLE}/complete.json" &&
   -f "${RQ1_EQUAL_REFERENCE}" ]] || {
  echo 'Private GOLD bundle or portable mono-source reference is incomplete.' >&2
  exit 1
}

export SAT_LODO_TEXT_ROOT="${text_root}"
# This campaign uses a fresh, isolated environment. Older .venv1 installations
# may have been partially deleted and must never be silently reused here.
export SAT_LODO_RUNTIME_ROOT="${storage_root}/safer_sat_equal_volume_runtime_v2"
export SAT_LODO_VENV="${SAT_LODO_RUNTIME_ROOT}/.venv"
export SAT_WTPSPLIT_REPO="${text_root}/vendor/wtpsplit"
export SAT_LODO_BUNDLE="${RQ1_EQUAL_SOURCE_BUNDLE}"
sat_lodo_set_paths
setup_hf_cache_env
if [[ "${1:-run}" == 'smoke' || "${1:-run}" == 'prepare' ]]; then
  sat_lodo_bootstrap
else
  sat_lodo_check_runtime
fi
source "${SAT_LODO_VENV}/bin/activate"
export PYTHONPATH="${text_root}"
export PYTHONNOUSERSITE=1 WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
echo "RQ1 equal-volume Python: $(command -v python)"
mkdir -p "${RQ1_EQUAL_RUN_ROOT}"
