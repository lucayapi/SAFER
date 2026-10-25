#!/usr/bin/env bash
# Submit smoke, M0-M2 baselines, six checkpoint-selected M3 folds (max 3 concurrent),
# then the final M0-M3 comparison. Run from SAFER/.
set -euo pipefail
smoke=$(sbatch --parsable text/jobs/sat_narrative_only_checkpoint_smoke.sh)
baseline=$(sbatch --parsable --dependency="afterok:${smoke}" text/jobs/sat_narrative_only_baselines.sh)
train=$(sbatch --parsable --dependency="afterok:${smoke}" text/jobs/sat_narrative_only_checkpoint_gpu.sh)
summary=$(sbatch --parsable --dependency="afterok:${baseline}:${train}" \
  text/jobs/sat_narrative_only_checkpoint_summary.sh)
printf 'smoke=%s\nbaseline_m0_m2=%s\nm3_array=%s\ncomparison=%s\n' \
  "${smoke}" "${baseline}" "${train}" "${summary}"
