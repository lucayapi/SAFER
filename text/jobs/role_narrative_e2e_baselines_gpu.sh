#!/usr/bin/env bash
#SBATCH --job-name=role-e2e-baselines
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/text/jobs/_role_qwen3_lodo_env.sh"
export OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
python -u scripts/role_narrative_e2e_baselines.py target --task-index "${SLURM_ARRAY_TASK_ID:?}"
