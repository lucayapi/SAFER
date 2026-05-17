# Helpers partagés par les jobs SLURM (sourcer depuis text/jobs/*.sh).
# Format attendu pour text/.env : KEY=valeur (voir .env.example), pas « KEY: valeur ».

# Charge text/.env dans l'environnement du job (set -a / source / set +a).
load_text_dotenv() {
  local env_file="${1:-.env}"
  if [[ ! -f "${env_file}" ]]; then
    return 0
  fi
  if grep -qE '^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*[[:space:]]*:' "${env_file}"; then
    echo "WARN: ${env_file} utilise « KEY: valeur » ; attendu KEY=value (voir .env.example). Ignoré." >&2
    return 0
  fi
  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a
}

# Active text/.venv si présent (depuis la racine text/, après cd ..).
activate_text_venv() {
  if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
  fi
}

# Racine text/ + .env + venv (à appeler après cd vers text/).
setup_text_job_env() {
  local env_file="${1:-.env}"
  load_text_dotenv "${env_file}"
  activate_text_venv
}
