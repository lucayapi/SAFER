#!/usr/bin/env bash
# Submit from text/: sbatch jobs/run_btp_qwen3_4b_instruct_leaf_umap_no_llm_factors.sh
# Uses existing role-specific instructed Qwen3-4B embeddings; runs factor discovery only.
# UMAP is enabled, HDBSCAN uses leaf, and factor annotation does not call an LLM.
# The four role CSVs must already exist in embeddings/instructed_qwen3_4b/btp_building_construction_and_masonry/.
#SBATCH --job-name=btp_qwen4b_umap_leaf_nollm
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

EMBEDDING_DIR="embeddings/instructed_qwen3_4b/btp_building_construction_and_masonry"
CONFIG="bayesian_network/config_btp_building_construction_masonry_qwen3_4b_instruct_leaf_umap_no_llm.yaml"

echo "[preflight] checking role-specific Qwen3-4B exports $(date -Iseconds)"
for role_file in \
  Qwen3-Embedding-4B_btp_B_mechanism_instruct.csv \
  Qwen3-Embedding-4B_btp_A0_work_situation_instruct.csv \
  Qwen3-Embedding-4B_btp_A1_adverse_condition_instruct.csv \
  Qwen3-Embedding-4B_btp_C_health_consequence_instruct.csv; do
  if [[ ! -s "${EMBEDDING_DIR}/${role_file}" ]]; then
    echo "Missing instructed embedding export: ${EMBEDDING_DIR}/${role_file}" >&2
    echo "Run jobs/export_all_roles_qwen3_4b_instruct.sh first, then submit this job." >&2
    exit 2
  fi
done

echo "[merge] validating and combining the four role exports $(date -Iseconds)"
python -u bayesian_network/merge_instructed_role_embeddings.py \
  --embedding-dir "${EMBEDDING_DIR}" \
  --model-prefix Qwen3-Embedding-4B

echo "[factor-discovery] UMAP enabled; HDBSCAN leaf; LLM annotation disabled $(date -Iseconds)"
python -u bayesian_network/run_bayesian_analysis.py \
  --config "${CONFIG}" \
  --stage factors \
  --reestimate

echo "[btp_qwen4b_umap_leaf_no_llm] factor discovery completed $(date -Iseconds)"
