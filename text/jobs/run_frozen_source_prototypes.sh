#!/bin/bash
# Frozen Source Prototypes — job unique paramétrable par BASE_METHOD.
#
# Usage:
#   cd ~/SAFER/text && bash jobs/run_frozen_source_prototypes.sh
#   BASE_METHOD=softtriple CORPUS=metallurgie bash jobs/run_frozen_source_prototypes.sh
#   BASE_METHOD=raw_embedding CORPUS=metallurgie bash jobs/run_frozen_source_prototypes.sh
#
# Variables:
#   CONFIG=configs/frozen_source_prototypes.yaml
#   CORPUS=metallurgie
#   BASE_METHOD=scgm_text
#   DEVICE=cuda

#SBATCH --job-name=fsp
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

CONFIG="${CONFIG:-configs/frozen_source_prototypes.yaml}"
CORPUS="${CORPUS:-metallurgie}"
BASE_METHOD="${BASE_METHOD:-scgm_text}"
DEVICE="${DEVICE:-cuda}"

echo "[fsp] CORPUS=${CORPUS} BASE_METHOD=${BASE_METHOD} DEVICE=${DEVICE} $(date -Iseconds)"
python -u scripts/run_frozen_source_prototypes.py \
  --config "${CONFIG}" \
  --corpus "${CORPUS}" \
  --base-method "${BASE_METHOD}"
echo "[fsp] terminé $(date -Iseconds)"
