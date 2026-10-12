#!/usr/bin/env bash
# From text/: sbatch jobs/submit_multisource_dg.sh
# This CPU job prepares source-only samples, then queues 126 GPU fits and analysis.
#SBATCH --job-name=msdg-submit
#SBATCH --partition=normal
#SBATCH --cpus-per-task=30
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
run_dir="${MSDG_RUN_DIR:-$PWD/output/multisource_dg/run_${SLURM_JOB_ID:?Submit with sbatch}}"
if [[ -e "$run_dir" ]]; then
  echo "Run directory already exists: $run_dir" >&2
  exit 2
fi
mkdir -p "$(dirname "$run_dir")"
python -u scripts/multisource_dg.py prepare --run-dir "$run_dir"
mkdir -p "$run_dir/logs"
run_dir="$(cd "$run_dir" && pwd)"
tasks="$(python scripts/multisource_dg.py count --run-dir "$run_dir")"
[[ "$tasks" =~ ^[0-9]+$ ]] && (( tasks > 0 )) || {
  echo "Invalid task count: $tasks" >&2
  exit 2
}
last=$((tasks - 1))
fits="$(sbatch --parsable --dependency="afterok:${SLURM_JOB_ID}" \
  --array="0-${last}%4" --export="ALL,MSDG_RUN_DIR=$run_dir" \
  --output="$run_dir/logs/fit-%A_%a.out" --error="$run_dir/logs/fit-%A_%a.err" \
  jobs/multisource_dg_gpu.sh)"
fits="${fits%%;*}"
[[ "$fits" =~ ^[0-9]+$ ]] || { echo "Invalid fit job ID: $fits" >&2; exit 2; }
analysis="$(sbatch --parsable --dependency="afterok:${fits}" \
  --export="ALL,MSDG_RUN_DIR=$run_dir" \
  --output="$run_dir/logs/analyze-%j.out" --error="$run_dir/logs/analyze-%j.err" \
  jobs/multisource_dg_analyze.sh)"
analysis="${analysis%%;*}"
[[ "$analysis" =~ ^[0-9]+$ ]] || { echo "Invalid analysis job ID: $analysis" >&2; exit 2; }
printf 'Run directory: %s\nGPU array: %s (%s tasks, max 4 simultaneous)\nAnalysis: %s\n' \
  "$run_dir" "$fits" "$tasks" "$analysis"
