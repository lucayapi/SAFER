#!/usr/bin/env bash
# From text/: bash jobs/submit_pgc_pilot.sh
# Submits dependency-linked source CV, final fits, and paired analysis.
set -euo pipefail
cd "$(dirname "$0")/.."

run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}"
run_dir="${PGC_RUN_DIR:-$PWD/output/pgc_pilot/run_${run_id}}"
mkdir -p "$run_dir/logs"
run_dir="$(cd "$run_dir" && pwd)"
if [[ -e "$run_dir/manifest.json" ]]; then
  echo "Run directory already contains a manifest: $run_dir" >&2
  exit 2
fi

common=(--parsable)
preflight="$(sbatch "${common[@]}" --export="ALL,PGC_RUN_DIR=$run_dir,PGC_PHASE=preflight" \
  --output="$run_dir/logs/preflight-%j.out" --error="$run_dir/logs/preflight-%j.err" jobs/pgc_pilot_cpu.sh)"
preflight="${preflight%%;*}"
[[ "$preflight" =~ ^[0-9]+$ ]] || { echo "Invalid preflight job ID: $preflight" >&2; exit 2; }
cv="$(sbatch "${common[@]}" --export="ALL,PGC_RUN_DIR=$run_dir,PGC_PHASE=cv" \
  --dependency="afterok:$preflight" --array=0-14%4 \
  --output="$run_dir/logs/cv-%A_%a.out" --error="$run_dir/logs/cv-%A_%a.err" jobs/pgc_pilot_gpu.sh)"
cv="${cv%%;*}"
[[ "$cv" =~ ^[0-9]+$ ]] || { echo "Invalid CV job ID: $cv" >&2; exit 2; }
selection="$(sbatch "${common[@]}" --export="ALL,PGC_RUN_DIR=$run_dir,PGC_PHASE=select" \
  --dependency="afterok:$cv" \
  --output="$run_dir/logs/select-%j.out" --error="$run_dir/logs/select-%j.err" jobs/pgc_pilot_cpu.sh)"
selection="${selection%%;*}"
[[ "$selection" =~ ^[0-9]+$ ]] || { echo "Invalid selection job ID: $selection" >&2; exit 2; }
final="$(sbatch "${common[@]}" --export="ALL,PGC_RUN_DIR=$run_dir,PGC_PHASE=final" \
  --dependency="afterok:$selection" --array=0-5%4 \
  --output="$run_dir/logs/final-%A_%a.out" --error="$run_dir/logs/final-%A_%a.err" jobs/pgc_pilot_gpu.sh)"
final="${final%%;*}"
[[ "$final" =~ ^[0-9]+$ ]] || { echo "Invalid final job ID: $final" >&2; exit 2; }
analysis="$(sbatch "${common[@]}" --export="ALL,PGC_RUN_DIR=$run_dir,PGC_PHASE=analyze" \
  --dependency="afterok:$final" \
  --output="$run_dir/logs/analyze-%j.out" --error="$run_dir/logs/analyze-%j.err" jobs/pgc_pilot_cpu.sh)"
analysis="${analysis%%;*}"

printf 'Run directory: %s\nPreflight: %s\nCV array: %s\nSelection: %s\nFinal array: %s\nAnalysis: %s\n' \
  "$run_dir" "$preflight" "$cv" "$selection" "$final" "$analysis"
