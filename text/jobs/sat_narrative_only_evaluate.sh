#!/usr/bin/env bash
# Re-evaluate already trained M3 adapters; no model training is repeated.
# Submit from SAFER/: sbatch text/jobs/sat_narrative_only_evaluate.sh
#SBATCH --job-name=sat-narrative-eval
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-5%3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/text/jobs/_sat_narrative_only_env.sh"
sat_narrative_only_setup run
: "${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}"
python -u scripts/sat_narrative_only_evaluate.py \
  --prepared "${SAT_NARRATIVE_PREPARED}" \
  --run-root "${SAT_NARRATIVE_RUN_ROOT}" \
  --wtpsplit-repo "${SAT_WTPSPLIT_REPO}" \
  --task-index "${SLURM_ARRAY_TASK_ID}"
