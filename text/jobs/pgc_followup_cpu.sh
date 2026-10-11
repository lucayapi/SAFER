#!/usr/bin/env bash
# Source-only setup/selection and target analysis.
#SBATCH --job-name=pgc-follow-cpu
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
: "${PGC_FOLLOWUP_PHASE:?}"
: "${PGC_FOLLOWUP_RUN_DIR:?}"
case "$PGC_FOLLOWUP_PHASE" in preflight|ce_analyze|supcon_select|supcon_analyze) ;;
  *) echo "Invalid CPU phase" >&2; exit 2 ;; esac
export OMP_NUM_THREADS=29 OPENBLAS_NUM_THREADS=29 MKL_NUM_THREADS=29
python -u scripts/run_pgc_followup.py "$PGC_FOLLOWUP_PHASE" \
  --config configs/pgc_followup.yaml --run-dir "$PGC_FOLLOWUP_RUN_DIR"
