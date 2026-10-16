#!/usr/bin/env bash
# Six rotations LoRA. Soumettre le smoke avant ce tableau.
# Depuis SAFER/ : sbatch text/jobs/sat_lodo_gpu.sh
# Depuis SAFER/text/ : sbatch jobs/sat_lodo_gpu.sh
#SBATCH --job-name=sat-lodo
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
if [[ -f "${SLURM_SUBMIT_DIR}/scripts/sat_lodo.py" ]]; then
  SAT_LODO_TEXT_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -f "${SLURM_SUBMIT_DIR}/text/scripts/sat_lodo.py" ]]; then
  SAT_LODO_TEXT_ROOT="${SLURM_SUBMIT_DIR}/text"
else
  echo "Soumettre depuis SAFER/ ou SAFER/text/." >&2
  exit 1
fi
cd "${SAT_LODO_TEXT_ROOT}"
source jobs/_env.sh
source jobs/_sat_lodo_runtime.sh
load_hpc_modules
sat_lodo_set_paths
setup_hf_cache_env

if [[ "${SAT_LODO_SMOKE:-0}" == "1" ]]; then
  sat_lodo_bootstrap
  command=smoke
else
  sat_lodo_check_runtime
  command=fit
fi
source "${SAT_LODO_VENV}/bin/activate"
export PYTHONPATH="${SAT_LODO_TEXT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
: "${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}"
mkdir -p "${SAT_LODO_RUN_ROOT}"
python -u scripts/sat_lodo.py check-bundle --bundle "${SAT_LODO_BUNDLE}"
python -u scripts/sat_lodo.py "${command}" \
  --bundle "${SAT_LODO_BUNDLE}" \
  --run-root "${SAT_LODO_RUN_ROOT}" \
  --wtpsplit-repo "${SAT_WTPSPLIT_REPO}" \
  --task-index "${SLURM_ARRAY_TASK_ID}"
