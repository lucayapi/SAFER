#!/usr/bin/env bash
# Summarize only after all six selected-checkpoint LODO tasks finish.
# From SAFER/: sbatch text/jobs/sat_lodo_gold_v3_checkpoint_summary.sh
#SBATCH --job-name=sat-v3-ckpt-summary
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
if [[ -f "${SLURM_SUBMIT_DIR}/text/jobs/_sat_lodo_gold_v3_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/text/jobs/_sat_lodo_gold_v3_env.sh"
else
  source "${SLURM_SUBMIT_DIR}/jobs/_sat_lodo_gold_v3_env.sh"
fi
export SAT_LODO_RUN_ROOT="${SCRATCH:-$HOME}/safer_sat_lodo/runs/sat12l_sm_gold_v3_rq1_checkpoint_selection_v1"
sat_gold_v3_setup
python -u scripts/sat_lodo_checkpoint_select.py summarize \
  --bundle "${SAT_LODO_BUNDLE}" \
  --run-root "${SAT_LODO_RUN_ROOT}"
