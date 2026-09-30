#!/bin/bash
# Soumet les 80 fits de la campagne 2 backbones x 2 sources x 4 mÃ©thodes x 5 seeds.
set -euo pipefail
CONFIG="${CONFIG:-output/replication_recipes/backbone_source_factorial.yaml}"
CONFIG="${CONFIG}" bash jobs/submit_replications.sh
