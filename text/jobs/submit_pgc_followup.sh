#!/usr/bin/env bash
# From text/: bash jobs/submit_pgc_followup.sh
# Submits all phases automatically. Target diagnostics never affect SupCon selection.
set -euo pipefail
cd "$(dirname "$0")/.."

run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}"
run_dir="${PGC_FOLLOWUP_RUN_DIR:-$PWD/output/pgc_followup/run_${run_id}}"
mkdir -p "$run_dir/logs"
run_dir="$(cd "$run_dir" && pwd)"
if [[ -e "$run_dir/manifest.json" ]]; then
  echo "Run directory already contains a manifest: $run_dir" >&2
  exit 2
fi

submit() {
  local id
  id="$(sbatch --parsable "$@")"
  id="${id%%;*}"
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "Invalid SLURM job ID: $id" >&2; exit 2; }
  printf '%s' "$id"
}

preflight="$(submit --export="ALL,PGC_FOLLOWUP_RUN_DIR=$run_dir,PGC_FOLLOWUP_PHASE=preflight" \
  --output="$run_dir/logs/preflight-%j.out" --error="$run_dir/logs/preflight-%j.err" jobs/pgc_followup_cpu.sh)"
ce="$(submit --dependency="afterok:$preflight" --array=0-14%4 \
  --export="ALL,PGC_FOLLOWUP_RUN_DIR=$run_dir,PGC_FOLLOWUP_PHASE=ce" \
  --output="$run_dir/logs/ce-%A_%a.out" --error="$run_dir/logs/ce-%A_%a.err" jobs/pgc_followup_gpu.sh)"
ce_analysis="$(submit --dependency="afterok:$ce" \
  --export="ALL,PGC_FOLLOWUP_RUN_DIR=$run_dir,PGC_FOLLOWUP_PHASE=ce_analyze" \
  --output="$run_dir/logs/ce-analysis-%j.out" --error="$run_dir/logs/ce-analysis-%j.err" jobs/pgc_followup_cpu.sh)"
supcon_cv="$(submit --dependency="afterok:$ce_analysis" --array=0-14%4 \
  --export="ALL,PGC_FOLLOWUP_RUN_DIR=$run_dir,PGC_FOLLOWUP_PHASE=supcon_cv" \
  --output="$run_dir/logs/supcon-cv-%A_%a.out" --error="$run_dir/logs/supcon-cv-%A_%a.err" jobs/pgc_followup_gpu.sh)"
selection="$(submit --dependency="afterok:$supcon_cv" \
  --export="ALL,PGC_FOLLOWUP_RUN_DIR=$run_dir,PGC_FOLLOWUP_PHASE=supcon_select" \
  --output="$run_dir/logs/supcon-select-%j.out" --error="$run_dir/logs/supcon-select-%j.err" jobs/pgc_followup_cpu.sh)"
supcon_final="$(submit --dependency="afterok:$selection" --array=0-5%4 \
  --export="ALL,PGC_FOLLOWUP_RUN_DIR=$run_dir,PGC_FOLLOWUP_PHASE=supcon_final" \
  --output="$run_dir/logs/supcon-final-%A_%a.out" --error="$run_dir/logs/supcon-final-%A_%a.err" jobs/pgc_followup_gpu.sh)"
supcon_analysis="$(submit --dependency="afterok:$supcon_final" \
  --export="ALL,PGC_FOLLOWUP_RUN_DIR=$run_dir,PGC_FOLLOWUP_PHASE=supcon_analyze" \
  --output="$run_dir/logs/supcon-analysis-%j.out" --error="$run_dir/logs/supcon-analysis-%j.err" jobs/pgc_followup_cpu.sh)"

printf 'Run directory: %s\nPreflight: %s\nCE diagnostic array: %s\nCE analysis: %s\nSupCon CV array: %s\nSupCon selection: %s\nSupCon final array: %s\nSupCon analysis: %s\n' \
  "$run_dir" "$preflight" "$ce" "$ce_analysis" "$supcon_cv" "$selection" "$supcon_final" "$supcon_analysis"
