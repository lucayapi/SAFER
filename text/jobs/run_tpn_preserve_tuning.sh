#!/bin/bash
# Balayage λ_pres (loss_weights.preserve) pour macro_transfer TPN.
#
# Usage :
#   cd ~/SAFER/text && bash jobs/run_tpn_preserve_tuning.sh
# Variables :
#   CORPUS=metallurgie
#   CONFIG=configs/tpn_macro_transfer.yaml
#   BASE_METHODS=scgm_text   # vide = tous les encodeurs
#   LAMBDA_PRES_GRID=0,0.05,0.10,0.25,0.50
#   SKIP_BERTOPIC=0
#   FORCE_REENCODE=0

#SBATCH --job-name=tpn_preserve_tune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
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
DEVICE="${DEVICE:-cuda}"
extra=(--config "${CONFIG}" --corpus "${CORPUS}" --device "${DEVICE}")
if [[ -n "${BASE_METHODS:-}" ]]; then
  extra+=(--base-methods "${BASE_METHODS}")
fi
if [[ -n "${LAMBDA_PRES_GRID:-}" ]]; then
  extra+=(--lambda-pres-grid "${LAMBDA_PRES_GRID}")
fi
if [[ "${SKIP_BERTOPIC:-0}" == "1" ]]; then
  extra+=(--skip-bertopic)
fi
if [[ "${FORCE_REENCODE:-0}" == "1" ]]; then
  extra+=(--force-reencode)
fi

echo "[tpn_preserve_tune] CORPUS=${CORPUS} $(date -Iseconds)"
python -u scripts/run_tpn_macro_transfer_preserve_tuning.py "${extra[@]}"
echo "[tpn_preserve_tune] terminé $(date -Iseconds)"
