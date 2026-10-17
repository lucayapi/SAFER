#!/usr/bin/env bash
# Queue the RQ1 LODO checkpoint-selection experiment with one sbatch command.
# From SAFER/: sbatch text/jobs/sat_lodo_gold_v3_checkpoint_launch.sh
#SBATCH --job-name=sat-v3-ckpt-launch
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

submit_dir="${SLURM_SUBMIT_DIR:?Submit from SAFER/ or SAFER/text/}"
if [[ -f "${submit_dir}/text/jobs/sat_lodo_gold_v3_checkpoint_smoke.sh" ]]; then
  jobs_dir="${submit_dir}/text/jobs"
elif [[ -f "${submit_dir}/jobs/sat_lodo_gold_v3_checkpoint_smoke.sh" ]]; then
  jobs_dir="${submit_dir}/jobs"
else
  echo "Submit from SAFER/ or SAFER/text/." >&2
  exit 1
fi

cd "${submit_dir}"
smoke_id="$(sbatch --parsable "${jobs_dir}/sat_lodo_gold_v3_checkpoint_smoke.sh")"
array_id="$(sbatch --parsable --dependency="afterok:${smoke_id}" \
  "${jobs_dir}/sat_lodo_gold_v3_checkpoint_gpu.sh")"
summary_id="$(sbatch --parsable --dependency="afterok:${array_id}" \
  "${jobs_dir}/sat_lodo_gold_v3_checkpoint_summary.sh")"
printf 'Smoke job: %s\nLODO GPU array: %s\nSummary job: %s\n' \
  "${smoke_id}" "${array_id}" "${summary_id}"
