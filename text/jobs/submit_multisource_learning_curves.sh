#!/usr/bin/env bash
# From text/: sbatch jobs/submit_multisource_learning_curves.sh
# Prepares samples, then queues GPU fits and final analysis via dependencies.
#SBATCH --job-name=mscurve-submit
#SBATCH --partition=normal
#SBATCH --cpus-per-task=30
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=29 OPENBLAS_NUM_THREADS=29 MKL_NUM_THREADS=29 NUMEXPR_NUM_THREADS=29
run_dir="${MSC_RUN_DIR:-$PWD/output/multisource_learning_curves/run_${SLURM_JOB_ID:?Submit with sbatch}}"
if [[ -e "$run_dir" ]]; then
  echo "Run directory already exists: $run_dir" >&2
  exit 2
fi
mkdir -p "$(dirname "$run_dir")"
python -u scripts/multisource_learning_curves.py prepare --run-dir "$run_dir"
mkdir -p "$run_dir/logs"
run_dir="$(cd "$run_dir" && pwd)"
tasks="$(python scripts/multisource_learning_curves.py count --run-dir "$run_dir")"
[[ "$tasks" =~ ^[0-9]+$ ]] && (( tasks > 0 )) || {
  echo "Invalid task count: $tasks" >&2
  exit 2
}
last=$((tasks - 1))
fits="$(sbatch --parsable --dependency="afterok:${SLURM_JOB_ID}" \
  --array="0-${last}%4" --export="ALL,MSC_RUN_DIR=$run_dir" \
  --output="$run_dir/logs/fit-%A_%a.out" --error="$run_dir/logs/fit-%A_%a.err" \
  jobs/multisource_learning_curves_gpu.sh)"
fits="${fits%%;*}"
[[ "$fits" =~ ^[0-9]+$ ]] || { echo "Invalid fit job ID: $fits" >&2; exit 2; }
analysis="$(sbatch --parsable --dependency="afterok:${fits}" \
  --export="ALL,MSC_RUN_DIR=$run_dir" \
  --output="$run_dir/logs/analyze-%j.out" --error="$run_dir/logs/analyze-%j.err" \
  jobs/multisource_learning_curves_analyze.sh)"
analysis="${analysis%%;*}"
[[ "$analysis" =~ ^[0-9]+$ ]] || { echo "Invalid analysis job ID: $analysis" >&2; exit 2; }
printf 'Run directory: %s\nGPU array: %s (%s tasks, max 4 simultaneous)\nAnalysis: %s\n' \
  "$run_dir" "$fits" "$tasks" "$analysis"
