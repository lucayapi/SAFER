#!/bin/bash
#SBATCH --job-name=supcon
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
_JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SLURM_SUBMIT_DIR:-$_JOB_DIR}/.."
# shellcheck source=jobs/_env.sh
source "${_JOB_DIR}/_env.sh"
setup_text_job_env

echo "HOST=$(hostname) DATE=$(date -Iseconds) JOB_ID=${SLURM_JOB_ID:-local}"

export HF_HOME="${SCRATCH:-$HOME}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"

python scripts/train_supcon.py --config configs/methods/supcon.yaml
