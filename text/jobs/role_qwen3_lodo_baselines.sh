#!/usr/bin/env bash
#SBATCH --job-name=role-qwen3-baselines
#SBATCH --partition=normal
#SBATCH --exclude=hpcnode39
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/text/jobs/_role_qwen3_lodo_env.sh"
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
python -u scripts/role_qwen3_lodo_campaign.py baseline
