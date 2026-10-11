#!/bin/bash
# Single submission command: sbatch jobs/submit_adapted_geometry.sh
#SBATCH --job-name=adapt-geo-submit
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
python -c 'import statsmodels.api' || {
  echo 'statsmodels missing in text/.venv; install text/requirements.txt before submission.' >&2
  exit 2
}
CONFIG="output/replication_recipes/adapted_geometry.yaml"
python scripts/prepare_adapted_geometry_campaign.py --output "${CONFIG}"
COUNT="$(python scripts/run_replication.py --config "${CONFIG}" --print-task-count)"
if [[ "${COUNT}" != "90" ]]; then
  echo "Expected 90 adapted refits, got ${COUNT}" >&2
  exit 2
fi
TRAIN_ID="$(sbatch --parsable --export=ALL,CONFIG="${CONFIG}" --array=0-89%4 jobs/adapted_geometry_refit.sh)"
FEATURES_ID="$(sbatch --parsable --dependency="afterok:${TRAIN_ID}" --export=ALL,CONFIG="${CONFIG}" --array=0-89%4 jobs/adapted_geometry_features.sh)"
SUMMARY_ID="$(sbatch --parsable --dependency="afterok:${FEATURES_ID}" --export=ALL,CONFIG="${CONFIG}" jobs/adapted_geometry_summary.sh)"
echo "Refits: ${TRAIN_ID} (90 tasks, up to 4 at once; 1 GPU + 30 CPU each)"
echo "Geometry: ${FEATURES_ID} (90 tasks, up to 4 at once; 30 CPU each)"
echo "Report: ${SUMMARY_ID} (starts after every geometry task succeeds)"
echo "Results: output/adapted_geometry/analysis/"
