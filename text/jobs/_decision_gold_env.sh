#!/usr/bin/env bash
set -euo pipefail
submit="${SLURM_SUBMIT_DIR:?Submit from SAFER/ or SAFER/text/}"
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
echo "decision-gold: $(python --version), submit=${submit}"
