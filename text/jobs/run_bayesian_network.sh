#!/usr/bin/env bash
# Configurable factor discovery, constrained BN learning, and scenario analysis.
# Submit from text/: CONFIG=bayesian_network/<config>.yaml sbatch jobs/run_bayesian_network.sh
# Optional: REESTIMATE=1, RUN_DIR=bayesian_network/runs/<existing-run-name>
#SBATCH --job-name=bayes_scenarios
#SBATCH --partition=normal
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SLURM_SUBMIT_DIR:-}/jobs/_bootstrap.sh" ]]; then source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
elif [[ -f "${SLURM_SUBMIT_DIR:-}/_bootstrap.sh" ]]; then source "${SLURM_SUBMIT_DIR}/_bootstrap.sh"
else source "${SCRIPT_DIR}/_bootstrap.sh"; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
ANALYSIS_CONFIG="${CONFIG:-bayesian_network/config.yaml}"
ARGS=(bayesian_network/run_bayesian_analysis.py --config "${ANALYSIS_CONFIG}")
if [[ "${REESTIMATE:-0}" == "1" ]]; then ARGS+=(--reestimate); fi
if [[ -n "${RUN_DIR:-}" ]]; then ARGS+=(--run-dir "${RUN_DIR}"); fi
echo "Running Bayesian scenario pipeline with config=${ANALYSIS_CONFIG}"
python -u "${ARGS[@]}"
