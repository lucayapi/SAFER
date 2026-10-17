#!/usr/bin/env bash
# CPU summary after all six narrative_v3 GPU tasks finish.
# From SAFER/: sbatch text/jobs/sat_lodo_gold_v3_summary.sh
#SBATCH --job-name=sat-gold-v3-summary
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
if [[ -f "${SLURM_SUBMIT_DIR}/text/jobs/_sat_lodo_gold_v3_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/text/jobs/_sat_lodo_gold_v3_env.sh"
else
  source "${SLURM_SUBMIT_DIR}/jobs/_sat_lodo_gold_v3_env.sh"
fi
sat_gold_v3_setup
python -u scripts/sat_lodo_gold_v3.py summarize \
  --bundle "${SAT_LODO_BUNDLE}" \
  --run-root "${SAT_LODO_RUN_ROOT}"
