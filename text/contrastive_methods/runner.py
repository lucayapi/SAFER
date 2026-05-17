"""Lance les scripts legacy contrastifs avec chemins resultats/."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from safer_core.io import flatten_method_config, load_yaml
from safer_core.paths import TEXT_ROOT, layout_method_output
from safer_core.text_columns import warn_if_prompt_enabled

METHOD_TO_LEGACY = {
    "batch_triplet": ("batchTripplet", "ftemb_script.py"),
    "softtriple": ("Softriple", "ftemb_script_softriple.py"),
    "supcon": ("Supcon", "ftemb_script_supcon.py"),
}


def run_contrastive_method(method_name: str, argv: list[str] | None = None) -> int:
    if method_name not in METHOD_TO_LEGACY:
        raise ValueError(method_name)
    legacy_folder, script_name = METHOD_TO_LEGACY[method_name]
    legacy_dir = TEXT_ROOT / "legacy" / "contrastive_method_v0" / legacy_folder
    script = legacy_dir / script_name
    if not script.is_file():
        raise FileNotFoundError(script)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=f"configs/methods/{method_name}.yaml")
    args, extra = parser.parse_known_args(argv)

    cfg = flatten_method_config(load_yaml(TEXT_ROOT / args.config))
    layout = layout_method_output(method_name, cfg.get("output_dir", f"resultats/{method_name}"))

    warn_if_prompt_enabled(bool(cfg.get("use_prompt", False)))

    data_csv = str(TEXT_ROOT / cfg.get("dataset_path", "dataset/data_btp.csv"))
    cmd = [
        sys.executable,
        str(script),
        "--input_csv",
        data_csv,
        "--target",
        str(cfg.get("label_col", "pred_label")),
        "--grid_output_root",
        str(layout["root"]),
        "--base_model_name",
        str(cfg.get("backbone_name", "Qwen/Qwen3-Embedding-0.6B")),
        "--use_contextual_prompt_with_summary",
        "false",
        "--use_fixed_instruction_prefix",
        "false",
    ]
    cmd.extend(extra)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(legacy_dir), str(TEXT_ROOT), env.get("PYTHONPATH", "")])
    print("Commande :", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=str(legacy_dir), env=env)
    if rc != 0:
        return rc

    from scripts.postprocess_contrastive_results import postprocess_contrastive_method

    try:
        postprocess_contrastive_method(method_name, args.config)
    except Exception as exc:
        print(f"[runner] Post-traitement échoué pour {method_name}: {exc}", flush=True)
        return 1
    return 0
