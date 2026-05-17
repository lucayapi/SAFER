#!/bin/bash
#SBATCH --job-name=softtriple
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=resultats/softtriple/logs/slurm-%x-%j.out
#SBATCH --error=resultats/softtriple/logs/slurm-%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}/.."
mkdir -p resultats/softtriple/logs

echo "HOST=$(hostname) DATE=$(date -Iseconds) JOB_ID=${SLURM_JOB_ID:-local}"

export HF_HOME="${SCRATCH:-$HOME}/hf_cache"
mkdir -p "${HF_HOME}"

python scripts/train_softtriple.py --config configs/methods/softtriple.yaml
