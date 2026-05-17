"""Tests lecture n_folds YAML → args.kfold pour SCGM."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from scripts.train_scgm_text import apply_config, parse_args


def test_apply_config_n_folds_maps_to_kfold():
    with patch.object(sys, "argv", ["train_scgm_text.py"]):
        args = parse_args()
    args.kfold = 0
    raw_path = TEXT_ROOT / "configs/scgm_text_strict_fidelity.yaml"
    apply_config(args, str(raw_path))
    assert args.kfold == 5
