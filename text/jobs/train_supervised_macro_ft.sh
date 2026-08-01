#!/bin/bash
# Fine-tuning supervised_macro_ft (CE + class_weight=balanced).
# Encodeur FT : last N layers ou full — pas d'encodeur gelé, pas d'oversampling.
#
# Usage:
#   cd ~/SAFER/text && sbatch jobs/train_supervised_macro_ft.sh
#   TRAIN_LAST_N_LAYERS=1 sbatch jobs/train_supervised_macro_ft.sh
#   TRAIN_LAST_N_LAYERS=null PROJECTION=null sbatch jobs/train_supervised_macro_ft.sh
#   TEST_CORPORA=metallurgie,caou,nicollin sbatch jobs/train_supervised_macro_ft.sh
#
# Campagne 8 variantes (tableau article) :
#   sbatch jobs/tune_supervised_macro_ft.sh

#SBATCH --job-name=sup-macro-ft
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
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
export TEST_CORPORA="${TEST_CORPORA:-metallurgie,caou,nicollin}"

ARGS=(--config "${CONFIG}" --test-corpora "${TEST_CORPORA}")
if [[ -n "${CLASS_WEIGHT:-}" ]]; then
  ARGS+=(--class-weight "${CLASS_WEIGHT}")
fi
if [[ -n "${TRAIN_LAST_N_LAYERS:-}" ]]; then
  ARGS+=(--train-last-n-layers "${TRAIN_LAST_N_LAYERS}")
fi
if [[ -n "${PROJECTION:-}" ]]; then
  ARGS+=(--projection "${PROJECTION}")
fi

echo "[sup-macro-ft] CONFIG=${CONFIG} TEST_CORPORA=${TEST_CORPORA} CLASS_WEIGHT=${CLASS_WEIGHT:-balanced} TRAIN_LAST_N_LAYERS=${TRAIN_LAST_N_LAYERS:-<yaml>} PROJECTION=${PROJECTION:-<yaml>} $(date -Iseconds)"
python -u scripts/train_supervised_macro_ft.py "${ARGS[@]}"
echo "[sup-macro-ft] terminé $(date -Iseconds)"
