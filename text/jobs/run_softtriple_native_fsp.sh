#!/bin/bash
# FSP SoftTriple — affectation par centres natifs W_{r,k} (job dédié).
#
# Usage:
#   cd ~/SAFER/text && bash jobs/run_softtriple_native_fsp.sh
#   CORPUS=metallurgie bash jobs/run_softtriple_native_fsp.sh
#
# Variables:
#   CONFIG=configs/frozen_source_prototypes_softtriple_native.yaml
#   CORPUS=metallurgie
#   DEVICE=cuda

#SBATCH --job-name=fsp-st-native
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

CONFIG="${CONFIG:-configs/frozen_source_prototypes_softtriple_native.yaml}"
CORPUS="${CORPUS:-metallurgie}"
DEVICE="${DEVICE:-cuda}"

echo "[fsp-st-native] CORPUS=${CORPUS} DEVICE=${DEVICE} $(date -Iseconds)"
python -u scripts/run_softtriple_native_fsp.py \
  --config "${CONFIG}" \
  --corpus "${CORPUS}"
echo "[fsp-st-native] terminé $(date -Iseconds)"
