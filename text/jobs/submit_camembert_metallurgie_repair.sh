#!/bin/bash
# Submit only the 15 adapted CamemBERT Metallurgy runs that must be repaired.
# Run from text/:
#   sbatch jobs/submit_camembert_metallurgie_repair.sh
# The valid Construction and frozen runs are not part of this recipe.
#SBATCH --job-name=camembert-met-repair-submit
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

CONFIG="${CONFIG:-output/replication_recipes/camembert_metallurgie_repair.yaml}"
TASK_COUNT="$(python scripts/run_replication.py --config "${CONFIG}" --print-task-count)"
if [[ "${TASK_COUNT}" != "15" ]]; then
  echo "Expected exactly 15 repair runs, got ${TASK_COUNT}." >&2
  exit 2
fi

TASKS="$(python scripts/run_replication.py --config "${CONFIG}" --print-tasks)"
if grep -q 'camembert_btp\|_frozen' <<<"${TASKS}"; then
  echo "Repair recipe unexpectedly contains Construction or frozen tasks." >&2
  exit 2
fi
if [[ "$(grep -c 'camembert_metallurgie_' <<<"${TASKS}")" != "15" ]]; then
  echo "Repair recipe does not contain exactly 15 Metallurgy tasks." >&2
  exit 2
fi

MAX_PARALLEL="${MAX_PARALLEL:-$(python scripts/run_replication.py --config "${CONFIG}" --print-max-parallel-seeds)}"
echo "Submitting only the 15 CamemBERT Metallurgy repairs (${MAX_PARALLEL} concurrent GPU jobs)."
echo "Outputs: output/camembert_metallurgie_repair"
ARRAY_SUBMISSION="$(sbatch --parsable --export=ALL,CONFIG="${CONFIG}" \
  --array="0-$((TASK_COUNT - 1))%${MAX_PARALLEL}" \
  "${TEXT_JOBS_DIR}/replicate_experiment.sh")"
ARRAY_JOB_ID="${ARRAY_SUBMISSION%%;*}"
echo "Repair array job: ${ARRAY_JOB_ID}"

ANALYSIS_OUTPUT="output/camembert_metallurgie_repair_analysis"
ANALYSIS_SUBMISSION="$(sbatch --parsable --dependency="afterok:${ARRAY_JOB_ID}" \
  --export=ALL,CONFIG="${CONFIG}",OUTPUT="${ANALYSIS_OUTPUT}" \
  "${TEXT_JOBS_DIR}/analyze_camembert_source_campaign.sh")"
ANALYSIS_JOB_ID="${ANALYSIS_SUBMISSION%%;*}"
echo "Dependent analysis job: ${ANALYSIS_JOB_ID}"
echo "Analysis outputs: ${ANALYSIS_OUTPUT}"
