#!/usr/bin/env bash
# One Qwen3 CE fit, one source mixture, one source draw, one training seed.
#SBATCH --job-name=msdg-fit
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=30
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
: "${MSDG_RUN_DIR:?MSDG_RUN_DIR is required}"
: "${SLURM_ARRAY_TASK_ID:?Submit as an array}"
export OMP_NUM_THREADS=29 OPENBLAS_NUM_THREADS=29 MKL_NUM_THREADS=29 NUMEXPR_NUM_THREADS=29
export TOKENIZERS_PARALLELISM=false
python -u scripts/multisource_dg.py fit \
  --run-dir "${MSDG_RUN_DIR}" --task-index "${SLURM_ARRAY_TASK_ID}"
