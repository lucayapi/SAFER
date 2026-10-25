#!/usr/bin/env bash
# From SAFER/: sbatch text/jobs/sat_narrative_only_summary.sh
#SBATCH --job-name=sat-narrative-summary
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/text/jobs/_sat_narrative_only_env.sh"
sat_narrative_only_setup run
python -u scripts/sat_narrative_only_lodo.py summarize \
  --prepared "${SAT_NARRATIVE_PREPARED}" \
  --run-root "${SAT_NARRATIVE_RUN_ROOT}"
