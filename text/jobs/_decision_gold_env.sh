#!/usr/bin/env bash
set -euo pipefail
submit="${SLURM_SUBMIT_DIR:?Submit from SAFER/ or SAFER/text/}"
# The decision-classification jobs use the shared text/.venv environment.
# Preserve an explicit cluster override when one is supplied.
export ROLE_GOLD_VENV="${ROLE_GOLD_VENV:-${SAFER_TEXT_VENV:-.venv}}"
if [[ -d "${submit}/text/jobs" ]]; then
  source "${submit}/text/jobs/_role_gold_env.sh"
elif [[ -d "${submit}/jobs" ]]; then
  source "${submit}/jobs/_role_gold_env.sh"
else
  echo 'Submit from SAFER/ or SAFER/text/.' >&2
  exit 2
fi
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
python -c 'import numpy, pandas, sklearn, torch, transformers; print("decision-gold dependencies:", "numpy=" + numpy.__version__, "torch=" + torch.__version__)'
echo "decision-gold: $(python --version), executable=$(command -v python), venv=${ROLE_GOLD_VENV}, submit=${submit}"
