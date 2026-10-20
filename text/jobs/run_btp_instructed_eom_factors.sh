#!/usr/bin/env bash
# Submit from text/: sbatch jobs/run_btp_instructed_eom_factors.sh
# Validates/combines the four instructed role exports, then runs factor discovery only.
# The four role CSVs must exist before submission.
#SBATCH --job-name=btp_instructed_eom
#SBATCH --partition=normal
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:?Submit this job from the text/ directory with sbatch}"
cd "${SUBMIT_DIR}"
source "${SUBMIT_DIR}/jobs/_bootstrap.sh"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1

echo "[instructed-embeddings] validating and combining role exports $(date -Iseconds)"
python -u bayesian_network/merge_instructed_role_embeddings.py

echo "[factor-discovery] UMAP/HDBSCAN with cluster_selection_method=eom $(date -Iseconds)"
python -u bayesian_network/run_bayesian_analysis.py \
  --config bayesian_network/config_btp_instructed_eom_r10.yaml \
  --stage factors \
  --reestimate

echo "[btp_instructed_eom] completed $(date -Iseconds)"
