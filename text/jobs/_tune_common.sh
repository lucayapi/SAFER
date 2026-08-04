# Helpers partagés pour jobs/tune_*.sh (sourcer depuis jobs/, pas sbatch directement).

tune_job_run() {
  local script="$1"
  local default_grid="$2"

  GRID_CONFIG="${GRID_CONFIG:-${default_grid}}"
  TEST_CORPUS="${TEST_CORPUS:-metallurgie}"
  export TEST_CORPUS

  local extra=(--grid-config "${GRID_CONFIG}")
  if [[ -n "${MAX_COMBOS:-}" ]]; then
    extra+=(--max-combos "${MAX_COMBOS}")
  fi
  if [[ "${SKIP_FINAL_FIT:-0}" == "1" ]]; then
    extra+=(--skip-final-fit)
  fi
  if [[ -n "${SEED:-}" ]]; then
    extra+=(--seed "${SEED}")
  fi

  echo "[tuning] script=${script} grid=${GRID_CONFIG} TEST_CORPUS=${TEST_CORPUS} MAX_COMBOS=${MAX_COMBOS:-all} SKIP_FINAL_FIT=${SKIP_FINAL_FIT:-0} SEED=${SEED:-default}"
  python "scripts/${script}" "${extra[@]}"
}

tune_architecture_job_run() {
  local script="$1"
  local default_grid="$2"

  GRID_CONFIG="${GRID_CONFIG:-${default_grid}}"
  local extra=(--grid-config "${GRID_CONFIG}")
  if [[ -n "${TEST_CORPORA:-}" ]]; then
    export TEST_CORPORA
    unset TEST_CORPUS
  elif [[ -n "${TEST_CORPUS:-}" ]]; then
    export TEST_CORPUS
  fi
  if [[ -n "${MAX_COMBOS:-}" ]]; then
    extra+=(--max-combos "${MAX_COMBOS}")
  fi
  if [[ "${SKIP_FINAL_FIT:-0}" == "1" ]]; then
    extra+=(--skip-final-fit)
  fi
  if [[ -n "${N_FOLDS:-}" ]]; then
    extra+=(--n-folds "${N_FOLDS}")
  fi
  if [[ -n "${SEED:-}" ]]; then
    extra+=(--seed "${SEED}")
  fi
  if [[ -n "${VARIANTS:-}" ]]; then
    read -r -a selected_variants <<< "${VARIANTS//,/ }"
    extra+=(--variants "${selected_variants[@]}")
  fi

  echo "[architecture-tuning] script=${script} grid=${GRID_CONFIG} VARIANTS=${VARIANTS:-all} TEST_CORPORA=${TEST_CORPORA:-${TEST_CORPUS:-config}} N_FOLDS=${N_FOLDS:-config}"
  python "scripts/${script}" "${extra[@]}"
}
