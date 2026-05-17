#!/bin/bash
# Soumission séquentielle (adapter partition/gpu selon le Mésocentre).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR/.."
mkdir -p resultats/raw_embedding/logs
mkdir -p resultats/scgm_text/logs
mkdir -p resultats/batch_triplet/logs
mkdir -p resultats/softtriple/logs
mkdir -p resultats/supcon/logs
mkdir -p resultats/comparisons/logs
cd "$DIR"

sbatch export_raw_embeddings.sh
sbatch train_scgm_text.sh
sbatch train_batch_triplet.sh
sbatch train_softtriple.sh
sbatch train_supcon.sh
echo "Jobs soumis. Suivi : squeue -u \$USER"
echo "Après complétion : sbatch compare_methods.sh"
