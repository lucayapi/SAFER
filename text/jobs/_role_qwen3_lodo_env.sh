#!/usr/bin/env bash
set -euo pipefail
submit="${SLURM_SUBMIT_DIR:?Submit from SAFER/}"
if [[ ! -d "${submit}/text/jobs" ]]; then
  echo 'Submit from the SAFER/ repository root.' >&2
  exit 2
fi
cd "${submit}/text"
source jobs/_env.sh
load_hpc_modules
load_text_dotenv .env
setup_hf_cache_env
if [[ ! -f .venv/bin/activate ]]; then
  echo 'Required Python environment text/.venv is missing.' >&2
  exit 2
fi
source .venv/bin/activate
if [[ "$(command -v python)" != "${submit}/text/.venv/bin/python" ]]; then
  echo "Wrong Python: $(command -v python); expected text/.venv/bin/python" >&2
  exit 2
fi
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
python -c 'import numpy, pandas, sklearn, torch, transformers; print("role-qwen3:", "numpy="+numpy.__version__, "torch="+torch.__version__)'
