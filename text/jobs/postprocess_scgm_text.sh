#!/bin/bash
# Post-traitement SCGM après train_scgm_text.sh :
#   1) embeddings Qwen test (si absents)
#   2) export SCGM complet BTP (topics, assignations, projected_embeddings.npy)
#   3) projections test (.npy pour notebook §8d)
#   4) enrichissement OpenAI (optionnel, SKIP_OPENAI=1 par défaut)
#
# Usage :
#   cd ~/SAFER/text && sbatch jobs/postprocess_scgm_text.sh
#   cd ~/SAFER/text && bash jobs/postprocess_scgm_text.sh
#
# OpenAI (login / Internet) :
#   SKIP_OPENAI=0 bash jobs/postprocess_scgm_text.sh
#
# Variables : SCGM_OUTPUT_DIR, CHECKPOINT, DATA_CSV, EMB_CSV, DATA_TEST_CSV,
#   EMB_TEST_CSV, BACKBONE_NAME, SKIP_TEST_EMB, SKIP_FULL_EXPORT, SKIP_TEST_PROJ,
#   SKIP_OPENAI, FORCE_EXPORT

#SBATCH --job-name=scgm_post
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/_bootstrap.sh" ]]; then
  # shellcheck source=jobs/_bootstrap.sh
  source "${SLURM_SUBMIT_DIR}/_bootstrap.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
else
  # shellcheck source=jobs/_bootstrap.sh
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

export HF_HOME="${SCRATCH:-$HOME}/hf_cache"
mkdir -p "${HF_HOME}"

SCGM_OUTPUT_DIR="${SCGM_OUTPUT_DIR:-resultats/scgm_text}"
CHECKPOINT="${CHECKPOINT:-${SCGM_OUTPUT_DIR}/checkpoints/best_model.pt}"
DATA_CSV="${DATA_CSV:-dataset/data_btp.csv}"
EMB_CSV="${EMB_CSV:-embeddings/Qwen3-Embedding-0.6B_btp.csv}"
DATA_TEST_CSV="${DATA_TEST_CSV:-dataset/test/data_metallurgie.csv}"
EMB_TEST_CSV="${EMB_TEST_CSV:-embeddings/Qwen3-Embedding-0.6B_metallurgie_test.csv}"
BACKBONE_NAME="${BACKBONE_NAME:-Qwen/Qwen3-Embedding-0.6B}"
SKIP_TEST_EMB="${SKIP_TEST_EMB:-0}"
SKIP_FULL_EXPORT="${SKIP_FULL_EXPORT:-0}"
SKIP_TEST_PROJ="${SKIP_TEST_PROJ:-0}"
SKIP_OPENAI="${SKIP_OPENAI:-1}"
FORCE_EXPORT="${FORCE_EXPORT:-0}"

THEMES_CSV="${SCGM_OUTPUT_DIR}/topics/themes_by_z.csv"
TEST_PROJ_NPY="${SCGM_OUTPUT_DIR}/embeddings/projected_embeddings_test.npy"

echo "HOST=$(hostname) DATE=$(date -Iseconds) JOB_ID=${SLURM_JOB_ID:-local}"
echo "SCGM_OUTPUT_DIR=${SCGM_OUTPUT_DIR}"
echo "CHECKPOINT=${CHECKPOINT}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[postprocess] ERREUR : checkpoint introuvable : ${CHECKPOINT}" >&2
  echo "  Lancez d'abord : sbatch jobs/train_scgm_text.sh" >&2
  exit 1
fi

# --- Étape 1/4 : embeddings Qwen test ---
if [[ "${SKIP_TEST_EMB}" == "1" ]]; then
  echo "[postprocess] étape 1/4 — SKIP_TEST_EMB=1 (embeddings test ignorés)"
elif [[ -f "${EMB_TEST_CSV}" ]]; then
  echo "[postprocess] étape 1/4 — déjà présent : ${EMB_TEST_CSV}"
else
  echo "[postprocess] étape 1/4 — export embeddings test (Qwen figé)…"
  python scripts/export_test_embeddings.py \
    --data_csv "${DATA_TEST_CSV}" \
    --output_csv "${EMB_TEST_CSV}" \
    --backbone_name "${BACKBONE_NAME}"
  echo "[postprocess]   → ${EMB_TEST_CSV}"
fi

# --- Étape 2/4 : export SCGM BTP complet ---
if [[ "${SKIP_FULL_EXPORT}" == "1" ]]; then
  echo "[postprocess] étape 2/4 — SKIP_FULL_EXPORT=1"
elif [[ "${FORCE_EXPORT}" != "1" && -f "${THEMES_CSV}" ]]; then
  echo "[postprocess] étape 2/4 — déjà présent : ${THEMES_CSV} (FORCE_EXPORT=1 pour régénérer)"
else
  echo "[postprocess] étape 2/4 — export SCGM BTP (topics, assignations, embeddings projetés)…"
  python scripts/export_scgm_text_outputs.py \
    --checkpoint "${CHECKPOINT}" \
    --output_dir "${SCGM_OUTPUT_DIR}" \
    --data_csv "${DATA_CSV}" \
    --emb_csv "${EMB_CSV}"
  echo "[postprocess]   → ${THEMES_CSV}"
  echo "[postprocess]   → ${SCGM_OUTPUT_DIR}/embeddings/projected_embeddings.npy"
fi

# --- Étape 3/4 : projections test SCGM ---
if [[ "${SKIP_TEST_PROJ}" == "1" ]]; then
  echo "[postprocess] étape 3/4 — SKIP_TEST_PROJ=1"
elif [[ ! -f "${EMB_TEST_CSV}" ]]; then
  echo "[postprocess] étape 3/4 — ignorée : ${EMB_TEST_CSV} absent" >&2
elif [[ "${FORCE_EXPORT}" != "1" && -f "${TEST_PROJ_NPY}" ]]; then
  echo "[postprocess] étape 3/4 — déjà présent : ${TEST_PROJ_NPY}"
else
  echo "[postprocess] étape 3/4 — projections test SCGM…"
  python scripts/export_scgm_test_projections.py \
    --checkpoint "${CHECKPOINT}" \
    --output_dir "${SCGM_OUTPUT_DIR}" \
    --data_csv "${DATA_TEST_CSV}" \
    --emb_csv "${EMB_TEST_CSV}"
  echo "[postprocess]   → ${TEST_PROJ_NPY}"
fi

# --- Étape 4/4 : OpenAI (optionnel) ---
if [[ "${SKIP_OPENAI}" == "1" ]]; then
  echo "[postprocess] étape 4/4 — SKIP_OPENAI=1 (thèmes OpenAI non générés)"
  echo "[postprocess] Terminé. Pour OpenAI : SKIP_OPENAI=0 bash jobs/postprocess_scgm_text.sh"
  exit 0
fi

if [[ ! -f "${THEMES_CSV}" ]]; then
  echo "[postprocess] étape 4/4 — ERREUR : ${THEMES_CSV} absent (étape 2 requise)" >&2
  exit 1
fi

echo "[postprocess] étape 4/4 — enrichissement OpenAI (themes_by_z → themes_by_z_openai)…"
OPENAI_ARGS=(
  --output_dir "${SCGM_OUTPUT_DIR}"
  --timeout "${OPENAI_TIMEOUT:-120}"
  --n-example-texts "${N_OPENAI_EXAMPLE_TEXTS:-5}"
  --model "${OPENAI_MODEL:-gpt-4o-mini}"
)
if [[ "${SKIP_ON_ERROR:-1}" == "1" ]]; then
  OPENAI_ARGS+=(--skip-on-error)
fi

set +e
python scripts/enrich_scgm_themes_openai.py "${OPENAI_ARGS[@]}"
openai_rc=$?
set -e

if [[ "${openai_rc}" -ne 0 ]]; then
  echo "[postprocess] AVERTISSEMENT : OpenAI a échoué (code ${openai_rc}). Exports GPU conservés." >&2
  echo "  Relancer sur le login : SKIP_OPENAI=0 bash jobs/enrich_scgm_themes_openai.sh" >&2
  exit "${openai_rc}"
fi

echo "[postprocess]   → ${SCGM_OUTPUT_DIR}/topics/themes_by_z_openai.csv"
echo "[postprocess] Terminé — notebook : python scripts/rebuild_notebook_01.py"
