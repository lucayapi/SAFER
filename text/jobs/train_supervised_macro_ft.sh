#!/bin/bash
# Entraînement supervised macro FT (CE, backbone Qwen + tête linear).
#
# Usage:
#   cd ~/SAFER/text && bash jobs/train_supervised_macro_ft.sh
#   STANDARDIZE_BACKBONE=true bash jobs/train_supervised_macro_ft.sh
#   OVERSAMPLING=true bash jobs/train_supervised_macro_ft.sh
#   BACKBONE_TRAINABLE=true TRAIN_LAST_N_LAYERS=4 bash jobs/train_supervised_macro_ft.sh
#   TEST_CORPORA=metallurgie,caou bash jobs/train_supervised_macro_ft.sh

#SBATCH --job-name=sup-macro-ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-user=lucayapi@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

CONFIG="${CONFIG:-configs/methods/supervised_macro_ft.yaml}"
export TEST_CORPORA="${TEST_CORPORA:-metallurgie,caou}"

ARGS=(--config "${CONFIG}" --test-corpora "${TEST_CORPORA}")
if [[ -n "${STANDARDIZE_BACKBONE:-}" ]]; then
  ARGS+=(--standardize-backbone "${STANDARDIZE_BACKBONE}")
fi
if [[ -n "${OVERSAMPLING:-}" ]]; then
  ARGS+=(--oversampling "${OVERSAMPLING}")
fi
if [[ -n "${CLASS_WEIGHT:-}" ]]; then
  ARGS+=(--class-weight "${CLASS_WEIGHT}")
fi
if [[ -n "${BACKBONE_TRAINABLE:-}" ]]; then
  ARGS+=(--backbone-trainable "${BACKBONE_TRAINABLE}")
fi
if [[ -n "${TRAIN_LAST_N_LAYERS:-}" ]]; then
  ARGS+=(--train-last-n-layers "${TRAIN_LAST_N_LAYERS}")
fi

echo "[sup-macro-ft] CONFIG=${CONFIG} TEST_CORPORA=${TEST_CORPORA} STANDARDIZE_BACKBONE=${STANDARDIZE_BACKBONE:-<yaml>} OVERSAMPLING=${OVERSAMPLING:-<yaml>} CLASS_WEIGHT=${CLASS_WEIGHT:-<yaml>} BACKBONE_TRAINABLE=${BACKBONE_TRAINABLE:-<yaml>} TRAIN_LAST_N_LAYERS=${TRAIN_LAST_N_LAYERS:-<yaml>} $(date -Iseconds)"
python -u scripts/train_supervised_macro_ft.py "${ARGS[@]}"
echo "[sup-macro-ft] terminé $(date -Iseconds)"
