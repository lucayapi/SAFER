#!/bin/bash
# Transfert macro TPN full-encoder (end-to-end) sur corpus test.
#
# Usage :
#   cd ~/SAFER/text && bash jobs/run_tpn_macro_transfer.sh
# Variables :
#   CORPUS=metallurgie
#   CONFIG=configs/tpn_macro_transfer.yaml   # config unique full encoder
#   BASE_METHOD=scgm_text      # scgm_text | softtriple | supcon | batch_triplet
#   CHECKPOINT=output/scgm_text_speed/checkpoints/best_model.pt
#   SKIP_BERTOPIC=1
#   DEVICE=cuda
#   EPOCHS=50
#   BACKBONE_NAME=Qwen/Qwen3-Embedding-0.6B
#   LR=2e-5
#   PROTOTYPE_MODE=batch  # batch | ema_global
#   PSEUDO_LABEL_THRESHOLD=0.6
#
# Exemples :
#   BASE_METHOD=softtriple bash jobs/run_tpn_macro_transfer.sh
#   CHECKPOINT=output/scgm_text_speed/checkpoints/best_model.pt bash jobs/run_tpn_macro_transfer.sh
#
# Prérequis : checkpoint entraîné pour l'encodeur choisi (train_*.sh)

#SBATCH --job-name=tpn_macro_transfer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
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

CONFIG="${CONFIG:-configs/tpn_macro_transfer.yaml}"
CORPUS="${CORPUS:-metallurgie}"
BASE_METHOD="${BASE_METHOD:-scgm_text}"
CHECKPOINT="${CHECKPOINT:-}"
SKIP_BERTOPIC="${SKIP_BERTOPIC:-1}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-5}"

export TEST_CORPUS="${CORPUS}"

extra=(--config "${CONFIG}" --corpus "${CORPUS}" --device "${DEVICE}" --epochs "${EPOCHS}")
if [[ -n "${BASE_METHOD}" ]]; then
  extra+=(--base-method "${BASE_METHOD}")
fi
if [[ -n "${CHECKPOINT}" ]]; then
  extra+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${BACKBONE_NAME:-}" ]]; then
  extra+=(--backbone-name "${BACKBONE_NAME}")
fi
if [[ -n "${LR:-}" ]]; then
  extra+=(--lr "${LR}")
fi
if [[ "${SKIP_BERTOPIC}" == "1" ]]; then
  extra+=(--skip-bertopic)
fi
if [[ -n "${PROTOTYPE_MODE:-}" ]]; then
  extra+=(--prototype-mode "${PROTOTYPE_MODE}")
fi
if [[ -n "${PSEUDO_LABEL_THRESHOLD:-}" ]]; then
  extra+=(--pseudo-label-threshold "${PSEUDO_LABEL_THRESHOLD}")
fi

echo "[tpn_macro_transfer] CORPUS=${CORPUS} BASE_METHOD=${BASE_METHOD} $(date -Iseconds)"
echo "[tpn_macro_transfer] Logs : tail -f slurm-${SLURM_JOB_ID:-local}.out  (PYTHONUNBUFFERED=1)"
# Entrypoint unique full encoder
python -u scripts/run_tpn_full_encoder_transfer.py "${extra[@]}"
echo "[tpn_macro_transfer] terminé CORPUS=${CORPUS} $(date -Iseconds)"
