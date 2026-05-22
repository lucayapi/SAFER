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
