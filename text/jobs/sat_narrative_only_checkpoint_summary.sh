#!/usr/bin/env bash
# Summarize selected M3 folds, then compare M0-M3 on their shared held-out narratives.
#SBATCH --job-name=sat-narr-ckpt-summary
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
export SAT_NARRATIVE_RUN_ROOT="${SAT_NARRATIVE_RUN_ROOT:-${SCRATCH:-$HOME}/safer_sat_lodo/runs/narrative_only_lodo_checkpoint_v1}"
source "${SLURM_SUBMIT_DIR}/text/jobs/_sat_narrative_only_env.sh"
sat_narrative_only_setup run
export SAT_NARRATIVE_BASELINE_ROOT="${SAT_NARRATIVE_BASELINE_ROOT:-${SCRATCH:-$HOME}/safer_sat_lodo/runs/narrative_only_lodo_v1_baselines}"
python -u scripts/sat_narrative_only_checkpoint_lodo.py summarize \
  --prepared "${SAT_NARRATIVE_PREPARED}" --run-root "${SAT_NARRATIVE_RUN_ROOT}"
python -u scripts/sat_narrative_only_baselines.py compare \
  --prepared "${SAT_NARRATIVE_PREPARED}" --baseline-root "${SAT_NARRATIVE_BASELINE_ROOT}" \
  --m3-root "${SAT_NARRATIVE_RUN_ROOT}" --output "${SAT_NARRATIVE_RUN_ROOT}/comparison"
