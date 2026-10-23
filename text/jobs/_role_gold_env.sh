#!/usr/bin/env bash
# Source from Slurm jobs copied to /tmp, never relative to BASH_SOURCE alone.
set -euo pipefail
submit="${SLURM_SUBMIT_DIR:?Submit from SAFER/}"
if [[ -d "${submit}/text/jobs" ]]; then
  cd "${submit}/text"
elif [[ -d "${submit}/jobs" ]]; then
  cd "${submit}"
else
  echo "Submit from SAFER/ or SAFER/text/" >&2
  exit 2
fi
source jobs/_env.sh
export PYTHONUNBUFFERED=1
load_hpc_modules
load_text_dotenv .env
setup_hf_cache_env
venv="${ROLE_GOLD_VENV:-${SAFER_TEXT_VENV:-.venv2}}"
if [[ ! -f "${venv}/bin/activate" ]]; then
  echo "Python environment missing: ${venv}" >&2
  exit 2
fi
source "${venv}/bin/activate"
export TOKENIZERS_PARALLELISM=false
echo "role-gold: $(python --version), venv=${venv}, submit=${submit}"
