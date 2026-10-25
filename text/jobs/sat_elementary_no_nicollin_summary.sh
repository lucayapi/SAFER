#!/usr/bin/env bash
#SBATCH --job-name=sat-no-nic-summary
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/text/jobs/_sat_elementary_no_nicollin_env.sh"
sat_elementary_no_nicollin_setup
python -u scripts/sat_elementary_no_nicollin.py summarize \
  --bundle "${SAT_LODO_BUNDLE}" --run-root "${SAT_NO_NICOLLIN_RUN_ROOT}"
