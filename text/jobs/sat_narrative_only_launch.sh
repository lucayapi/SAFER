#!/usr/bin/env bash
# Submit one smoke, six LODO GPU tasks (max 3 concurrent), then one summary.
set -euo pipefail
smoke=$(sbatch --parsable text/jobs/sat_narrative_only_smoke.sh)
train=$(sbatch --parsable --dependency="afterok:${smoke}" text/jobs/sat_narrative_only_gpu.sh)
summary=$(sbatch --parsable --dependency="afterok:${train}" text/jobs/sat_narrative_only_summary.sh)
printf 'smoke=%s\ntrain_array=%s\nsummary=%s\n' "${smoke}" "${train}" "${summary}"
