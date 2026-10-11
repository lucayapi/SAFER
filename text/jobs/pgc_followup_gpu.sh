#!/usr/bin/env bash
# One full-encoder fit per array task; 30 CPUs and one GPU.
#SBATCH --job-name=pgc-follow-gpu
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=30
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi
: "${PGC_FOLLOWUP_PHASE:?}"
: "${PGC_FOLLOWUP_RUN_DIR:?}"
: "${SLURM_ARRAY_TASK_ID:?}"
case "$PGC_FOLLOWUP_PHASE" in ce|supcon_cv|supcon_final) ;; *) echo "Invalid GPU phase" >&2; exit 2 ;; esac
# Use one fewer CPU thread than the 30 CPUs reserved for this task.
export OMP_NUM_THREADS=29 OPENBLAS_NUM_THREADS=29 MKL_NUM_THREADS=29
export TOKENIZERS_PARALLELISM=false
python -u scripts/run_pgc_followup.py "$PGC_FOLLOWUP_PHASE" \
  --config configs/pgc_followup.yaml --run-dir "$PGC_FOLLOWUP_RUN_DIR" \
  --task-id "$SLURM_ARRAY_TASK_ID"
