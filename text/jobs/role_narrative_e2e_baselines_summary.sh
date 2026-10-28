#!/usr/bin/env bash
#SBATCH --job-name=role-e2e-summary
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/text/jobs/_role_qwen3_lodo_env.sh"
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -u scripts/role_narrative_e2e_baselines.py summary
