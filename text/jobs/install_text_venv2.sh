#!/usr/bin/env bash
# Create/populate the shared text/.venv2 environment on the cluster.
# Submit from SAFER/: sbatch text/jobs/install_text_venv2.sh
# Safe to rerun: it does not remove an existing environment.
#SBATCH --job-name=safer-venv2
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail

submit="${SLURM_SUBMIT_DIR:?Submit from SAFER/}"
if [[ -d "${submit}/text/jobs" ]]; then
  cd "${submit}/text"
elif [[ -d "${submit}/jobs" && -f "${submit}/requirements.txt" ]]; then
  cd "${submit}"
else
  echo "Submit from SAFER/ or SAFER/text/" >&2
  exit 2
fi

source jobs/_env.sh
load_hpc_modules

venv=".venv2"
if [[ ! -f "${venv}/bin/activate" ]]; then
  echo "Creating ${PWD}/${venv}"
  python -m venv "${venv}"
fi
source "${venv}/bin/activate"

if [[ -n "${SCRATCH:-}" ]]; then
  export PIP_CACHE_DIR="${SCRATCH}/pip-cache"
  mkdir -p "${PIP_CACHE_DIR}"
fi

export INSTALL_TORCH_CUDA=1
bash scripts/install_text_venv.sh

python - <<'PY'
import numpy
import torch
import transformers

print(f"numpy={numpy.__version__}")
print(f"torch={torch.__version__}; CUDA build={torch.version.cuda}")
print(f"transformers={transformers.__version__}")
assert numpy.__version__.startswith("1.26"), "Expected numpy 1.26.x"
assert torch.version.cuda is not None, "Installed PyTorch is not a CUDA build"
assert transformers.__version__ == "4.51.3", "Unexpected transformers version"
PY

echo "Environment ready: ${PWD}/${venv}"
