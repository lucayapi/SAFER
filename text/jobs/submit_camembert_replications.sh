#!/bin/bash
# Apply source-only CamemBERT selections, then submit the 40 final runs
# (8 conditions x 5 seeds) as a Slurm job array.
# Submit only after all six source-only selection jobs have completed:
#   cd ~/SAFER/text
#   sbatch jobs/submit_camembert_replications.sh
#SBATCH --job-name=camembert-submit
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

CONFIG="${CONFIG:-output/replication_recipes/camembert_source_factorial.yaml}"
python scripts/apply_backbone_source_tuning.py \
  --recipe "${CONFIG}" \
  --tuning-root output/backbone_source_factorial/source_only_tuning \
  --write

TASK_COUNT="$(python scripts/run_replication.py --config "${CONFIG}" --print-task-count)"
if [[ "${TASK_COUNT}" != "40" ]]; then
  echo "Expected 40 CamemBERT runs, got ${TASK_COUNT}. Check n_seeds and models in ${CONFIG}." >&2
  exit 2
fi
MAX_PARALLEL="$(python scripts/run_replication.py --config "${CONFIG}" --print-max-parallel-seeds)"
echo "Submitting ${TASK_COUNT} CamemBERT runs from ${CONFIG} (${MAX_PARALLEL} concurrent seed jobs)."
sbatch --export=ALL,CONFIG="${CONFIG}" \
  --array="0-$((TASK_COUNT - 1))%${MAX_PARALLEL}" \
  "${TEXT_JOBS_DIR}/replicate_experiment.sh"
