#!/usr/bin/env bash
# Runs only after all array elements succeed.
#SBATCH --job-name=mscurve-analysis
#SBATCH --partition=normal
#SBATCH --cpus-per-task=30
#SBATCH --mem=96G
#SBATCH --time=06:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
: "${MSC_RUN_DIR:?MSC_RUN_DIR is required}"
export OMP_NUM_THREADS=29 OPENBLAS_NUM_THREADS=29 MKL_NUM_THREADS=29 NUMEXPR_NUM_THREADS=29
python -u scripts/multisource_learning_curves.py analyze --run-dir "${MSC_RUN_DIR}"
