#!/usr/bin/env bash
# Analysis starts only after every array element succeeds.
#SBATCH --job-name=msdg-analyze
#SBATCH --partition=normal
#SBATCH --cpus-per-task=30
#SBATCH --mem=96G
#SBATCH --time=06:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
: "${MSDG_RUN_DIR:?MSDG_RUN_DIR is required}"
export OMP_NUM_THREADS=29 OPENBLAS_NUM_THREADS=29 MKL_NUM_THREADS=29 NUMEXPR_NUM_THREADS=29
python -u scripts/multisource_dg.py analyze --run-dir "${MSDG_RUN_DIR}"
