"""Tests construction CLI runner contrastif."""

from __future__ import annotations

import sys
from pathlib import Path

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.runner import build_legacy_argv


def test_batch_triplet_argv_no_contextual_prompt():
    cfg = {
        "label_col": "pred_label",
        "backbone_name": "Qwen/Qwen3-Embedding-0.6B",
        "use_fixed_instruction_prefix": False,
        "use_contextual_prompt_with_summary": False,
    }
    argv = build_legacy_argv("batch_triplet", cfg, grid_output_root="/tmp/out", data_csv="data.csv")
    assert "--use_contextual_prompt_with_summary" not in argv
    assert "--use_fixed_instruction_prefix" in argv
    assert argv[-1] == "false"


def test_supcon_argv_with_contextual_prompt():
    cfg = {
        "label_col": "pred_label",
        "backbone_name": "Qwen/Qwen3-Embedding-0.6B",
        "use_fixed_instruction_prefix": False,
        "use_contextual_prompt_with_summary": False,
    }
    argv = build_legacy_argv("supcon", cfg, grid_output_root="/tmp/out", data_csv="data.csv")
    i = argv.index("--use_contextual_prompt_with_summary")
    assert argv[i + 1] == "false"
    i2 = argv.index("--use_fixed_instruction_prefix")
    assert argv[i2 + 1] == "false"


def test_softtriple_argv_with_contextual_prompt():
    cfg = {
        "label_col": "pred_label",
        "backbone_name": "Qwen/Qwen3-Embedding-0.6B",
        "use_fixed_instruction_prefix": True,
        "use_contextual_prompt_with_summary": False,
    }
    argv = build_legacy_argv("softtriple", cfg, grid_output_root="/tmp/out", data_csv="data.csv")
    assert "--use_contextual_prompt_with_summary" in argv
    i = argv.index("--use_fixed_instruction_prefix")
    assert argv[i + 1] == "true"
