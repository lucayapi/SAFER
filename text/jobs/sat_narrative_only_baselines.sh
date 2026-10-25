#!/usr/bin/env bash
# Evaluate M0-M2 on the same cleaned narrative-only LODO folds as M3.
# Submit from SAFER/: sbatch text/jobs/sat_narrative_only_baselines.sh
# Optional: export SAT_NARRATIVE_BASELINE_ROOT=/path/to/output before sbatch.
#SBATCH --job-name=sat-narrative-m0m2
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/text/jobs/_sat_narrative_only_env.sh"
sat_narrative_only_setup run
export SAT_NARRATIVE_BASELINE_ROOT="${SAT_NARRATIVE_BASELINE_ROOT:-${SCRATCH:-$HOME}/safer_sat_lodo/runs/narrative_only_lodo_v3_baselines}"
python -u scripts/sat_narrative_only_baselines.py run \
  --prepared "${SAT_NARRATIVE_PREPARED}" \
  --output "${SAT_NARRATIVE_BASELINE_ROOT}"
