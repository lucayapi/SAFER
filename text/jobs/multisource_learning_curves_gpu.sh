#!/usr/bin/env bash
# One Qwen3 CE fit for one total budget, source mixture, draw, and seed.
#SBATCH --job-name=mscurve-fit
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=30
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
: "${MSC_RUN_DIR:?MSC_RUN_DIR is required}"
: "${SLURM_ARRAY_TASK_ID:?Submit as an array}"
export OMP_NUM_THREADS=29 OPENBLAS_NUM_THREADS=29 MKL_NUM_THREADS=29 NUMEXPR_NUM_THREADS=29
export TOKENIZERS_PARALLELISM=false
python -u scripts/multisource_learning_curves.py fit \
  --run-dir "${MSC_RUN_DIR}" --task-index "${SLURM_ARRAY_TASK_ID}"
