#!/usr/bin/env bash
# A lancer avec sbatch apres la fin reussie des six taches GPU.
#SBATCH --job-name=sat-lodo-summary
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
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
sat_lodo_check_runtime
source "${SAT_LODO_VENV}/bin/activate"
python -u scripts/sat_lodo.py summarize --run-root "${SAT_LODO_RUN_ROOT}"
