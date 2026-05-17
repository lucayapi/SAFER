#!/bin/bash
#SBATCH --job-name=softriple_macro
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-user=lucayapi@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

echo "[START] $(date '+%F %T')"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module purge
module load gcc/8.1.0
module load python/3.10.10
module load cuda/12.0.1

# =========================================================
# VALEURS PAR DÉFAUT
# =========================================================
INPUT_CSV="data/Qwen_Qwen3-8B__v1__snapshot.csv"
TARGET="pred_label"
GRID_OUTPUT_ROOT="models_research/grid_search_softtriple_embeddinggemma-300m_pred_label"
BASE_MODEL_NAME="google/embeddinggemma-300m"

USE_CONTEXTUAL_PROMPT_WITH_SUMMARY="false"
SUMMARY_COL="accident_summary"

USE_FIXED_INSTRUCTION_PREFIX="false"
FIXED_INSTRUCTION_PREFIX="Represent this occupational accident factual unit according to its prevention-relevant role in the accident scenario."

# =========================================================
# PARSING DES ARGUMENTS
# Exemple :
# sbatch jobft_softriple.sh \
#   --target pred_label \
#   --grid_output_root models_research/grid_search_softtriple_qwen_06_pred_label \
#   --base_model_name Qwen/Qwen3-Embedding-0.6B \
#   --use_fixed_instruction_prefix true
# =========================================================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_csv)
      INPUT_CSV="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    --grid_output_root)
      GRID_OUTPUT_ROOT="$2"
      shift 2
      ;;
    --base_model_name)
      BASE_MODEL_NAME="$2"
      shift 2
      ;;
    --use_contextual_prompt_with_summary)
      USE_CONTEXTUAL_PROMPT_WITH_SUMMARY="$2"
      shift 2
      ;;
    --summary_col)
      SUMMARY_COL="$2"
      shift 2
      ;;
    --use_fixed_instruction_prefix)
      USE_FIXED_INSTRUCTION_PREFIX="$2"
      shift 2
      ;;
    --fixed_instruction_prefix)
      FIXED_INSTRUCTION_PREFIX="$2"
      shift 2
      ;;
    *)
      echo "[ERROR] Argument inconnu : $1"
      exit 1
      ;;
  esac
done

if [[ "$TARGET" != "pred_label" && "$TARGET" != "pred_subtype" ]]; then
  echo "[ERROR] --target doit être 'pred_label' ou 'pred_subtype'. Reçu: $TARGET"
  exit 1
fi

if [[ -z "$GRID_OUTPUT_ROOT" ]]; then
  GRID_OUTPUT_ROOT="models_research/grid_search_softtriple_qwen_06_${TARGET}"
fi

echo "[INFO] INPUT_CSV=$INPUT_CSV"
echo "[INFO] TARGET=$TARGET"
echo "[INFO] GRID_OUTPUT_ROOT=$GRID_OUTPUT_ROOT"
echo "[INFO] BASE_MODEL_NAME=$BASE_MODEL_NAME"
echo "[INFO] USE_CONTEXTUAL_PROMPT_WITH_SUMMARY=$USE_CONTEXTUAL_PROMPT_WITH_SUMMARY"
echo "[INFO] SUMMARY_COL=$SUMMARY_COL"
echo "[INFO] USE_FIXED_INSTRUCTION_PREFIX=$USE_FIXED_INSTRUCTION_PREFIX"
echo "[INFO] FIXED_INSTRUCTION_PREFIX=$FIXED_INSTRUCTION_PREFIX"

# Active l'environnement virtuel
[ -f "env-ftemb/bin/activate" ] || { echo "[ERROR] env-ftemb/bin/activate introuvable"; exit 1; }
source "env-ftemb/bin/activate"

# Vérifie que les scripts existent
[ -f "ftemb_script_softriple.py" ] || { echo "[ERROR] ftemb_script_softriple.py introuvable"; exit 1; }
[ -f "ftemb_softtriple.py" ] || { echo "[ERROR] ftemb_softtriple.py introuvable"; exit 1; }

# Cache HF sur scratch
SCRATCH_BASE="${SCRATCH:-/storage/scratch/$USER}"
export HF_HOME="${SCRATCH_BASE}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export SENTENCE_TRANSFORMERS_HOME="${HF_HOME}/sentence_transformers"
mkdir -p "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$SENTENCE_TRANSFORMERS_HOME"

# Réduit les warnings / surcharges CPU inutiles
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Infos GPU
nvidia-smi || true

# Lance le fine-tuning + encodage
srun python -u ftemb_script_softriple.py \
  --input_csv "$INPUT_CSV" \
  --target "$TARGET" \
  --grid_output_root "$GRID_OUTPUT_ROOT" \
  --base_model_name "$BASE_MODEL_NAME" \
  --use_contextual_prompt_with_summary "$USE_CONTEXTUAL_PROMPT_WITH_SUMMARY" \
  --summary_col "$SUMMARY_COL" \
  --use_fixed_instruction_prefix "$USE_FIXED_INSTRUCTION_PREFIX" \
  --fixed_instruction_prefix "$FIXED_INSTRUCTION_PREFIX"

echo "[END] $(date '+%F %T')"