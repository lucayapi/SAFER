#!/usr/bin/env bash
# Source-only setup/selection and read-only target analysis.
#SBATCH --job-name=pgc-cpu
#SBATCH --partition=normal
#SBATCH --cpus-per-task=30
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

: "${PGC_PHASE:?PGC_PHASE is required}"
: "${PGC_RUN_DIR:?PGC_RUN_DIR is required}"
case "$PGC_PHASE" in preflight|select|analyze) ;; *) echo "Invalid CPU phase: $PGC_PHASE" >&2; exit 2 ;; esac
export OMP_NUM_THREADS=29
export OPENBLAS_NUM_THREADS=29
export MKL_NUM_THREADS=29
python -u scripts/run_pgc_pilot.py "$PGC_PHASE" --config configs/pgc_pilot.yaml --run-dir "$PGC_RUN_DIR"
