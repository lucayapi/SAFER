#!/bin/bash
#SBATCH --job-name=raw_emb
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
_JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SLURM_SUBMIT_DIR:-$_JOB_DIR}/.."
# shellcheck source=jobs/_env.sh
source "${_JOB_DIR}/_env.sh"
setup_text_job_env

export HF_HOME="${SCRATCH:-$HOME}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"

python scripts/export_raw_embeddings.py --config configs/methods/raw_embedding.yaml
