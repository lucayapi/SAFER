#!/bin/bash
# From text/: sbatch jobs/export_diagnostic_camembert_embeddings.sh
# Four independent corpus exports can run concurrently if GPUs are available.
#SBATCH --job-name=diag-cam-emb
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=30
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3%4
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

CORPORA=(btp metallurgie caou nicollin)
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch}"
if (( TASK_INDEX < 0 || TASK_INDEX >= ${#CORPORA[@]} )); then
  echo "Invalid array task index: ${TASK_INDEX}" >&2
  exit 2
fi
CORPUS="${CORPORA[TASK_INDEX]}"
THREADS=$((SLURM_CPUS_PER_TASK - 1))
export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export TOKENIZERS_PARALLELISM=false

echo "host=$(hostname) job=${SLURM_JOB_ID} task=${TASK_INDEX} corpus=${CORPUS} threads=${THREADS}"
ARGS=(--corpus "${CORPUS}")
if [[ "${FORCE:-0}" == "1" ]]; then
  ARGS+=(--force)
fi
python -u scripts/export_diagnostic_camembert_embeddings.py "${ARGS[@]}"
