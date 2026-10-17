#!/usr/bin/env bash
# Submit the complete six-corpus experiment with one sbatch command.
# From SAFER/: sbatch text/jobs/sat_indomain_v3_launch.sh
#SBATCH --job-name=sat-indomain-launch
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

submit_dir="${SLURM_SUBMIT_DIR:?Submit this script with sbatch from SAFER or SAFER/text}"
if [[ -f "${submit_dir}/text/jobs/_sat_lodo_gold_v3_env.sh" ]]; then
  source "${submit_dir}/text/jobs/_sat_lodo_gold_v3_env.sh"
  jobs_dir="${submit_dir}/text/jobs"
elif [[ -f "${submit_dir}/jobs/_sat_lodo_gold_v3_env.sh" ]]; then
  source "${submit_dir}/jobs/_sat_lodo_gold_v3_env.sh"
  jobs_dir="${submit_dir}/jobs"
else
  echo "Lancer depuis SAFER/ ou SAFER/text/." >&2
  exit 1
fi

# This job prepares the private data before submitting any GPU work.
unset SAT_LODO_BUNDLE
sat_gold_v3_setup
private_root="${SCRATCH:-$HOME}/safer_sat_indomain"
bundle="${private_root}/bundle/sat12l_sm_gold_v3_indomain_v1"
if [[ ! -f "${bundle}/complete.json" ]]; then
  python -u scripts/sat_indomain_v3.py prepare \
    --source-bundle "${SAT_LODO_BUNDLE}" --out "${bundle}"
fi
python -u scripts/sat_indomain_v3.py check-bundle --bundle "${bundle}"

# Submit from the original directory so downstream logs stay together.
cd "${submit_dir}"
smoke_id="$(sbatch --parsable "${jobs_dir}/sat_indomain_v3_smoke.sh")"
array_id="$(sbatch --parsable --dependency="afterok:${smoke_id}" \
  "${jobs_dir}/sat_indomain_v3_gpu.sh")"
summary_id="$(sbatch --parsable --dependency="afterok:${array_id}" \
  "${jobs_dir}/sat_indomain_v3_summary.sh")"
printf 'Bundle: %s\nSmoke job: %s\nGPU array: %s\nSummary job: %s\n' \
  "${bundle}" "${smoke_id}" "${array_id}" "${summary_id}"
