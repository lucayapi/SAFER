#!/bin/bash
# Soumission séquentielle (adapter partition/gpu selon le Mésocentre).
# Lancer depuis text/jobs/ : cd ~/SAFER/text/jobs && bash submit_all.sh
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Logs SLURM : slurm-<job_name>-<job_id>.out|.err dans ce dossier (jobs/).
# Les jobs résolvent text/ via SLURM_SUBMIT_DIR + jobs/_bootstrap.sh (pas /tmp/slurmd/...).

sbatch export_raw_geometry.sh
TRAIN_SCGM_ID=$(sbatch --parsable train_scgm_text.sh)
echo "train_scgm_text job_id=${TRAIN_SCGM_ID}"
sbatch train_batch_triplet.sh
sbatch train_softtriple.sh
sbatch train_supcon.sh
echo "Jobs soumis. Suivi : squeue -u \$USER"
echo "Après train SCGM : sbatch export_test_embeddings.sh  # si CSV test absent"
echo "Puis : BASE_METHOD=scgm_text CORPUS=metallurgie bash run_frozen_source_prototypes.sh"
echo "Comparaisons : sbatch compare_methods.sh"
