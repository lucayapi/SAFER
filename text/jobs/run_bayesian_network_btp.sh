#!/usr/bin/env bash
# Full BTP Bayesian network analysis: discovery -> LLM names -> matrix -> BN -> scenarios.
# Submit from text/: sbatch jobs/run_bayesian_network_btp.sh
# Optional: REESTIMATE=1 sbatch jobs/run_bayesian_network_btp.sh
# Optional: RUN_DIR=bayesian_network/runs/my_run sbatch jobs/run_bayesian_network_btp.sh
#SBATCH --job-name=btp_bayesian_network
#SBATCH --partition=normal
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SLURM_SUBMIT_DIR:-}/jobs/_bootstrap.sh" ]]; then source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
elif [[ -f "${SLURM_SUBMIT_DIR:-}/_bootstrap.sh" ]]; then source "${SLURM_SUBMIT_DIR}/_bootstrap.sh"
else source "${SCRIPT_DIR}/_bootstrap.sh"; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
ARGS=(bayesian_network/run_bayesian_analysis.py --config bayesian_network/config.yaml)
if [[ "${REESTIMATE:-0}" == "1" ]]; then ARGS+=(--reestimate); fi
if [[ -n "${RUN_DIR:-}" ]]; then ARGS+=(--run-dir "${RUN_DIR}"); fi
echo "Running BTP full-corpus Bayesian analysis; embeddings=${PWD}/embeddings/gold_v3/e5/btp.csv"
python -u "${ARGS[@]}"
