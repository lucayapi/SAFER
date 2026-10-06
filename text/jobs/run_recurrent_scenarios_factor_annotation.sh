#!/usr/bin/env bash
# Resume factor naming without rerunning discovery (for a job submitted before
# factor_annotation was integrated into STAGE=all, or after an API interruption).
# Usage from text/: DATASET=btp_carpentry_and_joinery CONFIG_PATH=... RUN_DIR=... sbatch jobs/run_recurrent_scenarios_factor_annotation.sh
#SBATCH --job-name=accident_factor_labels
#SBATCH --partition=normal
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
DATASET="${DATASET:-btp_carpentry_and_joinery}"
CONFIG_PATH="${CONFIG_PATH:-recurrent_scenarios/config_jrssc_rebuild.yaml}"
RUN_DIR="${RUN_DIR:-recurrent_scenarios/runs/theme_discovery_rebuild/${DATASET}}"

python -u recurrent_scenarios/factor_annotation.py \
  --config "${CONFIG_PATH}" --dataset "${DATASET}" --run-dir "${RUN_DIR}"
