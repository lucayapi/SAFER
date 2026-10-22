#!/usr/bin/env bash
# From text/: sbatch jobs/run_curated_tdm_bn.sh
# Recode the completed leaf run, then fit the strict and sensitivity Bayesian
# networks and the K=3/K=1 TDM analyses. No embeddings, UMAP, HDBSCAN or LLM
# annotation are rerun. A completed unchanged curation is reused.
#SBATCH --job-name=curated_tdm_bn
#SBATCH --partition=normal
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SLURM_SUBMIT_DIR:-}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
elif [[ -f "${SLURM_SUBMIT_DIR:-}/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/_bootstrap.sh"
else
  source "${SCRIPT_DIR}/_bootstrap.sh"
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
LEAF_RUN="${LEAF_RUN:-bayesian_network/runs/btp_building_construction_masonry_qwen3_4b_raw_leaf_umap_no_llm_20261022T004957Z}"
CURATED_RUN="${CURATED_RUN:-bayesian_network/runs/curated_leaf_components_qwen3_4b}"
SOURCE_NAME="$(basename "${CURATED_RUN%/}")"
BN_OUTPUT="${BN_OUTPUT:-bayesian_network/runs/${SOURCE_NAME}_network}"
TDM_CONFIG="${TDM_CONFIG:-theme_dictionary/config_curated_leaf.yaml}"
TDM_SELECTION="${TDM_SELECTION:-theme_dictionary/runs/curated_leaf_selection}"
echo "[curated] leaf=${LEAF_RUN}; output=${CURATED_RUN}; CPUs=${SLURM_CPUS_PER_TASK:-24}"
echo "[curated] unit-level component recoding"
python -u bayesian_network/curate_leaf_components.py \
  --source-run "${LEAF_RUN}" \
  --output-dir "${CURATED_RUN}"
echo "[curated] finding the smallest feasible declared TDM support threshold"
python -u -m theme_dictionary.select_feasible_config \
  --config "${TDM_CONFIG}" --source-run "${CURATED_RUN}" \
  --output-dir "${TDM_SELECTION}"
TDM_SELECTED_CONFIG="${TDM_SELECTION}/config_selected.yaml"
echo "[curated] Bayesian networks and conditional bootstrap"
python -u bayesian_network/run_network_from_matrix.py \
  --source-run "${CURATED_RUN}" \
  --config bayesian_network/config_btp_building_construction_masonry_qwen3_4b_raw_leaf_umap_no_llm.yaml \
  --output-dir "${BN_OUTPUT}"
echo "[curated] TDM estimation, K=3 and K=1"
python -u -m theme_dictionary.run \
  --config "${TDM_SELECTED_CONFIG}" --source-run "${CURATED_RUN}"
echo "[curated] all analyses complete"
