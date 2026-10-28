#!/usr/bin/env bash
#SBATCH --job-name=role-e2e-smoke
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/text/jobs/_role_qwen3_lodo_env.sh"
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -u scripts/role_narrative_e2e_baselines.py smoke
