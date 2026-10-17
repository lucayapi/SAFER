#!/usr/bin/env bash
# Source-validation threshold check; no target test inference or re-training.
#SBATCH --job-name=sat-lodo-low-thr
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

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
sat_lodo_check_runtime
source "${SAT_LODO_VENV}/bin/activate"

# Only v3 source validation files and the original prediction caches are read.
SAT_LODO_ARTICLE_RESULTS="${SAT_LODO_RUN_ROOT%/}/../resultats_articles"
python -u scripts/sat_lodo_low_thresholds.py run \
  --bundle "${SAT_LODO_BUNDLE}" \
  --runs "${SAT_LODO_RUN_ROOT}" \
  --out "${SAT_LODO_ARTICLE_RESULTS}/validation_seuils_inferieurs"
