#!/bin/bash
# Runs only after all 90 geometry tasks succeeded.
#SBATCH --job-name=adapt-geo-report
#SBATCH --partition=normal
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --cpus-per-task=30
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
export OMP_NUM_THREADS=29 OPENBLAS_NUM_THREADS=29 MKL_NUM_THREADS=29 NUMEXPR_NUM_THREADS=29
CONFIG="${CONFIG:-output/replication_recipes/adapted_geometry.yaml}"
python -u scripts/summarize_adapted_geometry.py --config "${CONFIG}"
