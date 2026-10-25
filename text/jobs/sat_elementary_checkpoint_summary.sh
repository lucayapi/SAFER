#!/usr/bin/env bash
# From SAFER/: sbatch text/jobs/sat_elementary_checkpoint_summary.sh
#SBATCH --job-name=sat-elem-ckpt-summary
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/text/jobs/_sat_elementary_checkpoint_env.sh"
sat_elementary_checkpoint_setup
python -u scripts/sat_elementary_checkpoint_lodo.py summarize \
  --bundle "${SAT_LODO_BUNDLE}" --prepared "${SAT_ELEMENTARY_PREPARED}" \
  --run-root "${SAT_ELEMENTARY_CHECKPOINT_RUN_ROOT}"
