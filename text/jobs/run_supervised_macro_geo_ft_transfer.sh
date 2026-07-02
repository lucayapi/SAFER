#!/bin/bash
# Transfert test + BERTopic pour supervised_macro_geo_ft.
#
# Usage:
#   CORPUS=metallurgie bash jobs/run_supervised_macro_geo_ft_transfer.sh
#   RUN_BERTOPIC=false CORPUS=metallurgie bash jobs/run_supervised_macro_geo_ft_transfer.sh
#   JUDGE_ENABLE=false CORPUS=metallurgie bash jobs/run_supervised_macro_geo_ft_transfer.sh

#SBATCH --job-name=sup-macro-geo-xfer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

CONFIG="${CONFIG:-configs/supervised_macro_geo_ft_transfer.yaml}"
CORPUS="${CORPUS:-metallurgie}"
ARGS=(--config "${CONFIG}" --corpus "${CORPUS}")
if [[ "${RUN_BERTOPIC:-true}" == "false" ]]; then
  ARGS+=(--skip-bertopic)
fi
if [[ -n "${JUDGE_ENABLE:-}" ]]; then
  ARGS+=(--judge-enable "${JUDGE_ENABLE}")
fi

echo "[sup-macro-geo-xfer] CORPUS=${CORPUS} RUN_BERTOPIC=${RUN_BERTOPIC:-true} JUDGE_ENABLE=${JUDGE_ENABLE:-<yaml>} $(date -Iseconds)"
python -u scripts/run_supervised_macro_ft_transfer.py "${ARGS[@]}"
echo "[sup-macro-geo-xfer] terminé $(date -Iseconds)"
