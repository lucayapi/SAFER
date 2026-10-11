#!/bin/bash
# One refit x seed; exact adapted-space source neighbours and role margin.
#SBATCH --job-name=adapt-geo-nn
#SBATCH --partition=normal
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --cpus-per-task=30
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
export OMP_NUM_THREADS=29 OPENBLAS_NUM_THREADS=29 MKL_NUM_THREADS=29 NUMEXPR_NUM_THREADS=29
CONFIG="${CONFIG:-output/replication_recipes/adapted_geometry.yaml}"
python -u scripts/analyze_adapted_geometry_run.py --config "${CONFIG}" --task-index "${SLURM_ARRAY_TASK_ID:?}"
