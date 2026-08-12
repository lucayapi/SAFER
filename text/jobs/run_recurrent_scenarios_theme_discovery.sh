#!/usr/bin/env bash
# Parallel D_U/S_R theme discovery for the recurrent-accident analysis.
#
# Usage from text/:
#   REESTIMATE=1 sbatch jobs/run_recurrent_scenarios_theme_discovery.sh
#   STAGE=metrics REESTIMATE=1 sbatch jobs/run_recurrent_scenarios_theme_discovery.sh
#   STAGE=pareto RUN_DIR=recurrent_scenarios/runs/audit_caou sbatch jobs/run_recurrent_scenarios_theme_discovery.sh

#SBATCH --job-name=accident_themes
#SBATCH --partition=normal
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-user=lucayapi@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/_bootstrap.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/text/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/text/jobs/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

# One thread per worker; the Python config resolves workers to allocated cores - 1.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Leave the dataset unset unless the job explicitly receives DATASET=...
# so that config.yaml's data.dataset_id remains the default source of truth.
DATASET="${DATASET:-}"
CONFIG_PATH="${CONFIG_PATH:-recurrent_scenarios/config.yaml}"
REESTIMATE="${REESTIMATE:-0}"
DEBUG="${DEBUG:-0}"
STAGE="${STAGE:-all}"
RUN_DIR="${RUN_DIR:-}"

ARGS=(recurrent_scenarios/run_theme_discovery.py --config "${CONFIG_PATH}" --stage "${STAGE}")
if [[ -n "${DATASET}" ]]; then
  ARGS+=(--dataset "${DATASET}")
fi
if [[ "${REESTIMATE}" == "1" ]]; then
  ARGS+=(--reestimate)
fi
if [[ "${DEBUG}" == "1" ]]; then
  ARGS+=(--debug)
fi
if [[ -n "${RUN_DIR}" ]]; then
  ARGS+=(--run-dir "${RUN_DIR}")
fi

echo "Dataset=${DATASET:-from-config}"
echo "Config=${CONFIG_PATH}"
echo "Stage=${STAGE}"
echo "RunDir=${RUN_DIR:-auto}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "Launching: python -u ${ARGS[*]}"
python -u "${ARGS[@]}"
