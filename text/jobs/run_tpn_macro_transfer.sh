#!/bin/bash
# Transfert macro TPN (encodeur gelé modulable + adaptateur) sur corpus test.
#
# Usage :
#   cd ~/SAFER/text && bash jobs/run_tpn_macro_transfer.sh
# Variables :
#   CORPUS=metallurgie
#   CONFIG=configs/tpn_macro_transfer.yaml
#   BASE_METHOD=softtriple   # softtriple | supcon | batch_triplet | scgm_text
#   CHECKPOINT=output/softtriple/checkpoints/best_model
#   SKIP_BERTOPIC=0
#   DEVICE=cuda
#   EPOCHS=50
#
# Exemples :
#   BASE_METHOD=supcon CONFIG=configs/tpn_macro_transfer_supcon.yaml bash jobs/run_tpn_macro_transfer.sh
#   BASE_METHOD=scgm_text CONFIG=configs/tpn_macro_transfer_scgm.yaml bash jobs/run_tpn_macro_transfer.sh
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

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

CONFIG="${CONFIG:-configs/tpn_macro_transfer.yaml}"
CORPUS="${CORPUS:-metallurgie}"
CHECKPOINT="${CHECKPOINT:-}"
SKIP_BERTOPIC="${SKIP_BERTOPIC:-0}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-50}"

export TEST_CORPUS="${CORPUS}"

extra=(--config "${CONFIG}" --corpus "${CORPUS}" --device "${DEVICE}" --epochs "${EPOCHS}")
if [[ -n "${BASE_METHOD}" ]]; then
  extra+=(--base-method "${BASE_METHOD}")
fi
if [[ -n "${CHECKPOINT}" ]]; then
  extra+=(--checkpoint "${CHECKPOINT}")
fi
if [[ "${SKIP_BERTOPIC}" == "1" ]]; then
  extra+=(--skip-bertopic)
fi

echo "[tpn_macro_transfer] CORPUS=${CORPUS} $(date -Iseconds)"
python scripts/run_tpn_macro_transfer_discovery.py "${extra[@]}"
echo "[tpn_macro_transfer] terminé CORPUS=${CORPUS} $(date -Iseconds)"
