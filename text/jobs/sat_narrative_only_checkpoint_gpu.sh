#!/usr/bin/env bash
# Six shared-test LODO folds; one seed, at most three concurrent GPU tasks.
#SBATCH --job-name=sat-narr-ckpt
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
export SAT_NARRATIVE_RUN_ROOT="${SAT_NARRATIVE_RUN_ROOT:-${SCRATCH:-$HOME}/safer_sat_lodo/runs/narrative_only_lodo_checkpoint_v1}"
source "${SLURM_SUBMIT_DIR}/text/jobs/_sat_narrative_only_env.sh"
sat_narrative_only_setup run
: "${SLURM_ARRAY_TASK_ID:?Submit this script as an array}"
python -u scripts/sat_narrative_only_checkpoint_lodo.py fit \
  --prepared "${SAT_NARRATIVE_PREPARED}" --run-root "${SAT_NARRATIVE_RUN_ROOT}" \
  --wtpsplit-repo "${SAT_WTPSPLIT_REPO}" --task-index "${SLURM_ARRAY_TASK_ID}"
