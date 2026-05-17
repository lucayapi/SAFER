# Bootstrap jobs SLURM : cd vers text/ + .env + venv.
# SLURM copie le .sh dans /tmp — utiliser SLURM_SUBMIT_DIR pour sourcer ce fichier,
# pas dirname(BASH_SOURCE) du script job copié.

TEXT_JOBS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEXT_ROOT="$(cd "${TEXT_JOBS_DIR}/.." && pwd)"
cd "${TEXT_ROOT}"
# shellcheck source=jobs/_env.sh
source "${TEXT_JOBS_DIR}/_env.sh"
setup_text_job_env
