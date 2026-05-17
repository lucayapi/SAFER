#!/bin/bash
#SBATCH --job-name=raw_emb
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}/.."

python scripts/export_raw_embeddings.py --config configs/methods/raw_embedding.yaml
