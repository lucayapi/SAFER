# Helpers partagés par les jobs SLURM (sourcer depuis text/jobs/*.sh).
# Charge text/.env sans « source » bash (compatible KEY=value et KEY: valeur).

load_text_dotenv() {
  local env_file="${1:-.env}"
  if [[ ! -f "${env_file}" ]]; then
    return 0
  fi

  local line trimmed key val
  while IFS= read -r line || [[ -n "${line}" ]]; do
    # Commentaires (# en début de ligne après espaces)
    trimmed="${line#"${line%%[![:space:]]*}"}"
    [[ -z "${trimmed}" || "${trimmed}" == \#* ]] && continue

    if [[ "${trimmed}" =~ ^([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*)$ ]]; then
      key="${BASH_REMATCH[1]}"
      val="${BASH_REMATCH[2]}"
    elif [[ "${trimmed}" =~ ^([A-Za-z_][A-Za-z0-9_]*):[[:space:]]*(.*)$ ]]; then
      key="${BASH_REMATCH[1]}"
      val="${BASH_REMATCH[2]}"
      echo "WARN: ${env_file} — préférer ${key}=… (pas ${key}: …)" >&2
    else
      echo "WARN: ${env_file} — ligne ignorée : ${trimmed}" >&2
      continue
    fi

    # Espaces en bord de valeur
    val="${val#"${val%%[![:space:]]*}"}"
    val="${val%"${val##*[![:space:]]}"}"
    # Retirer guillemets optionnels
    if [[ "${val}" =~ ^\"(.*)\"$ ]]; then
      val="${BASH_REMATCH[1]}"
    elif [[ "${val}" =~ ^\'(.*)\'$ ]]; then
      val="${BASH_REMATCH[1]}"
    fi
    export "${key}=${val}"
  done < "${env_file}"
}

# Modules Environment (Mésocentre) — avant activation du venv.
load_hpc_modules() {
  if [[ "${SAFER_SKIP_MODULES:-0}" == "1" ]]; then
    return 0
  fi
  if ! command -v module >/dev/null 2>&1; then
    return 0
  fi
  local gcc_mod="${SAFER_GCC_MODULE:-gcc/8.1.0}"
  local py_mod="${SAFER_PYTHON_MODULE:-python/3.10.10}"
  module purge
  module load "${gcc_mod}"
  module load "${py_mod}"
  echo "modules: gcc=${gcc_mod} python=${py_mod} ($(python --version 2>&1))"
}

# Active text/.venv si présent (depuis la racine text/, après cd ..).
activate_text_venv() {
  if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
  fi
}

# Racine text/ + modules HPC + .env + venv (à appeler après cd vers text/).
setup_text_job_env() {
  local env_file="${1:-.env}"
  load_hpc_modules
  load_text_dotenv "${env_file}"
  activate_text_venv
}
