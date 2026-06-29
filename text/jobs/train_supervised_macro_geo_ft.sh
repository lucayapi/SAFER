#!/bin/bash
# Entraînement supervised macro geo FT (CE + λ·L_geo).
#
# Usage:
#   cd ~/SAFER/text && bash jobs/train_supervised_macro_geo_ft.sh
#   LAMBDA_GEO=0.05 bash jobs/train_supervised_macro_geo_ft.sh
#   TEST_CORPUS=metallurgie bash jobs/train_supervised_macro_geo_ft.sh
#
# Sweep λ suggéré : 0.01, 0.05, 0.1, 0.5, 1.0

#SBATCH --job-name=sup-macro-geo-ft
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

CONFIG="${CONFIG:-configs/methods/supervised_macro_geo_ft.yaml}"
export TEST_CORPUS="${TEST_CORPUS:-metallurgie}"

ARGS=(--config "${CONFIG}")
if [[ -n "${LAMBDA_GEO:-}" ]]; then
  ARGS+=(--lambda-geo "${LAMBDA_GEO}")
fi

echo "[sup-macro-geo-ft] CONFIG=${CONFIG} LAMBDA_GEO=${LAMBDA_GEO:-<yaml>} TEST_CORPUS=${TEST_CORPUS} $(date -Iseconds)"
python -u scripts/train_supervised_macro_ft.py "${ARGS[@]}"
echo "[sup-macro-geo-ft] terminé $(date -Iseconds)"
