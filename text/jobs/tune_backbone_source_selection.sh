#!/bin/bash
# Une sÃ©lection source-only : une mÃ©thode Ã— backbone Ã— corpus source.
# Le YAML fourni porte test_corpora: [], donc aucune cible OOD n'est lue.
# Exemple :
#   TUNING_METHOD=supcon GRID_CONFIG=output/.../qwen3_metallurgie_supcon_grid.yaml \
#     sbatch jobs/tune_backbone_source_selection.sh

#SBATCH --job-name=source_select
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-user=lucayapi@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

: "${TUNING_METHOD:?TUNING_METHOD requis: cross_entropy, supcon ou softtriple}"
: "${GRID_CONFIG:?GRID_CONFIG requis}"
case "${TUNING_METHOD}" in
  cross_entropy) SCRIPT="scripts/tune_supervised_macro_ft.py" ;;
  supcon) SCRIPT="scripts/tune_supcon_macro_ft.py" ;;
  softtriple) SCRIPT="scripts/tune_softtriple_macro_ft.py" ;;
  *) echo "TUNING_METHOD invalide: ${TUNING_METHOD}" >&2; exit 2 ;;
esac

echo "HOST=$(hostname) DATE=$(date -Iseconds) METHOD=${TUNING_METHOD} GRID=${GRID_CONFIG}"
python -u "${SCRIPT}" --grid-config "${GRID_CONFIG}" --skip-final-fit --refit "${REFIT:-true}"
