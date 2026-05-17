"""Lance les scripts legacy contrastifs avec chemins resultats/."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from safer_core.io import flatten_method_config, load_yaml
from safer_core.paths import TEXT_ROOT, layout_method_output
from safer_core.text_columns import warn_if_prompt_enabled

METHOD_TO_LEGACY = {
    "batch_triplet": ("batchTripplet", "ftemb_script.py"),
    "softtriple": ("Softriple", "ftemb_script_softriple.py"),
    "supcon": ("Supcon", "ftemb_script_supcon.py"),
}

# Scripts legacy qui acceptent --use_contextual_prompt_with_summary
_METHODS_WITH_CONTEXTUAL_PROMPT = frozenset({"softtriple", "supcon"})


def _bool_cli(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y", "oui", "on"}:
        return "true"
    return "false"


def build_legacy_argv(
    method_name: str,
    cfg: dict[str, Any],
    *,
    grid_output_root: str | Path,
    data_csv: str | Path,
) -> list[str]:
    """Arguments CLI passés au ftemb_script*.py (hors exécutable et chemin script)."""
    if method_name not in METHOD_TO_LEGACY:
        raise ValueError(method_name)

    argv = [
        "--input_csv",
        str(data_csv),
        "--target",
        str(cfg.get("label_col", "pred_label")),
        "--grid_output_root",
        str(grid_output_root),
        "--base_model_name",
        str(cfg.get("backbone_name", "Qwen/Qwen3-Embedding-0.6B")),
        "--use_fixed_instruction_prefix",
        _bool_cli(cfg.get("use_fixed_instruction_prefix", False)),
    ]
    if method_name in _METHODS_WITH_CONTEXTUAL_PROMPT:
        argv.extend(
            [
                "--use_contextual_prompt_with_summary",
                _bool_cli(cfg.get("use_contextual_prompt_with_summary", False)),
            ]
        )
    return argv


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

    data_csv = TEXT_ROOT / cfg.get("dataset_path", "dataset/data_btp.csv")
    cmd = [
        sys.executable,
        str(script),
        *build_legacy_argv(
            method_name,
            cfg,
            grid_output_root=layout["root"],
            data_csv=data_csv,
        ),
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
