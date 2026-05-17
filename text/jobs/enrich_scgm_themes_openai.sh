#!/bin/bash
# Enrichissement OpenAI des thèmes SCGM (themes_by_z → themes_by_z_openai).
#
# Nécessite un accès Internet vers api.openai.com (ou OPENAI_BASE_URL dans text/.env).
# Sur HPC2 : préférer l'exécution sur le nœud de LOGIN, pas via JupyterHub / GPU :
#   cd ~/SAFER/text && bash jobs/enrich_scgm_themes_openai.sh
#
# Si la partition normal a accès sortant, vous pouvez aussi :
#   sbatch jobs/enrich_scgm_themes_openai.sh

#SBATCH --job-name=scgm_openai_themes
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}/.."

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Ne pas « source .env » : format shell KEY=value uniquement (pas KEY: valeur).
# La clé est chargée par python-dotenv dans scripts/enrich_scgm_themes_openai.py

OUTPUT_DIR="${SCGM_OUTPUT_DIR:-resultats/scgm_text}"
PROBE_ONLY="${PROBE_ONLY:-0}"

ARGS=(
  --output_dir "${OUTPUT_DIR}"
  --timeout "${OPENAI_TIMEOUT:-120}"
  --n-example-texts "${N_OPENAI_EXAMPLE_TEXTS:-5}"
  --model "${OPENAI_MODEL:-gpt-4o-mini}"
)

if [[ "${PROBE_ONLY}" == "1" ]]; then
  ARGS+=(--probe-only)
fi

if [[ "${SKIP_ON_ERROR:-1}" == "1" ]]; then
  ARGS+=(--skip-on-error)
fi

echo "HOST=$(hostname) DATE=$(date -Iseconds) OUTPUT_DIR=${OUTPUT_DIR}"
python scripts/enrich_scgm_themes_openai.py "${ARGS[@]}"
