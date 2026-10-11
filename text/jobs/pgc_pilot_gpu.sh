#!/usr/bin/env bash
# One Qwen3 fine-tuning fit per array task.
#SBATCH --job-name=pgc-gpu
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

: "${PGC_PHASE:?PGC_PHASE is required}"
: "${PGC_RUN_DIR:?PGC_RUN_DIR is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
case "$PGC_PHASE" in cv|final) ;; *) echo "Invalid GPU phase: $PGC_PHASE" >&2; exit 2 ;; esac
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
python -u scripts/run_pgc_pilot.py "$PGC_PHASE" --config configs/pgc_pilot.yaml \
  --run-dir "$PGC_RUN_DIR" --task-id "$SLURM_ARRAY_TASK_ID"
