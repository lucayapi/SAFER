#!/bin/bash
# Transfère les neuf sélections source-only vers la recette de campagne finale.
# À lancer depuis text/ : sbatch jobs/apply_backbone_source_tuning.sh

#SBATCH --job-name=apply_source_selection
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

echo "HOST=$(hostname) DATE=$(date -Iseconds)"
python -u scripts/apply_backbone_source_tuning.py --write
