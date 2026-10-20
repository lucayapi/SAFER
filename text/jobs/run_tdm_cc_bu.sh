#!/usr/bin/env bash
# From text/: sbatch jobs/run_tdm_cc_bu.sh
# Independent fits: sbatch --array=1,3 jobs/run_tdm_cc_bu.sh
# Or submit separate jobs with TDM_K=3 and TDM_K=1.
# Optional: CONFIG=theme_dictionary/other.yaml RUN_DIR=theme_dictionary/runs/name
# Optional preflight: PREFLIGHT=1 sbatch jobs/run_tdm_cc_bu.sh
#SBATCH --job-name=tdm_cc_bu
#SBATCH --partition=normal
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SLURM_SUBMIT_DIR:-}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
elif [[ -f "${SLURM_SUBMIT_DIR:-}/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/_bootstrap.sh"
else
  source "${SCRIPT_DIR}/_bootstrap.sh"
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
CONFIG_PATH="${CONFIG:-theme_dictionary/config.yaml}"
ARGS=(--config "${CONFIG_PATH}")
if [[ -n "${RUN_DIR:-}" ]]; then ARGS+=(--run-dir "${RUN_DIR}"); fi
SELECTED_K="${TDM_K:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -n "${SELECTED_K}" ]]; then ARGS+=(--k "${SELECTED_K}"); fi
if [[ "${PREFLIGHT:-0}" == "1" ]]; then ARGS+=(--preflight); fi
echo "TDM-CC-BU-K: config=${CONFIG_PATH}, K=${SELECTED_K:-3,1}, CPUs=${SLURM_CPUS_PER_TASK:-24}"
python -u -m theme_dictionary.run "${ARGS[@]}"
