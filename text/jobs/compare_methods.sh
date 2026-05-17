#!/bin/bash
#SBATCH --job-name=compare_emb
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}/.."

python scripts/collect_results.py --results_root resultats
python scripts/compare_methods.py --results_root resultats --output_dir resultats/comparisons
