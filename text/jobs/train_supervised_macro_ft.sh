#!/bin/bash
# Entraînement supervised macro FT (CE, backbone Qwen + tête linear).
#
# Usage:
#   cd ~/SAFER/text && bash jobs/train_supervised_macro_ft.sh
#   STANDARDIZE_BACKBONE=true bash jobs/train_supervised_macro_ft.sh
#   TEST_CORPUS=metallurgie bash jobs/train_supervised_macro_ft.sh

#SBATCH --job-name=sup-macro-ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

CONFIG="${CONFIG:-configs/methods/supervised_macro_ft.yaml}"
export TEST_CORPUS="${TEST_CORPUS:-metallurgie}"

ARGS=(--config "${CONFIG}")
if [[ -n "${STANDARDIZE_BACKBONE:-}" ]]; then
  ARGS+=(--standardize-backbone "${STANDARDIZE_BACKBONE}")
fi

echo "[sup-macro-ft] CONFIG=${CONFIG} TEST_CORPUS=${TEST_CORPUS} STANDARDIZE_BACKBONE=${STANDARDIZE_BACKBONE:-<yaml>} $(date -Iseconds)"
python -u scripts/train_supervised_macro_ft.py "${ARGS[@]}"
echo "[sup-macro-ft] terminé $(date -Iseconds)"
