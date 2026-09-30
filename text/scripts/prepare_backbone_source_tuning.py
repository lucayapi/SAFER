"""Materialise source-only tuning configs for the factorial campaign.

The generated configs set ``test_corpora: []``.  Consequently the existing
tuning programs can never read a target corpus during model selection.  Run
this once, run the 12 commands written to ``commands.sh``, inspect each
``grid_summary.csv``, then transfer the selected recipe to the final campaign.
"""

from __future__ import annotations

import copy
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {"btp": "dataset/data_btp.csv", "metallurgie": "dataset/data_metallurgie.csv"}
BACKBONES = {
    "qwen3": ("Qwen/Qwen3-Embedding-0.6B", ""),
    "multilingual_e5_large": ("intfloat/multilingual-e5-large", "query: "),
}


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-qwen3-btp", action="store_true", help="RecrÃ©er les trois sÃ©lections dÃ©jÃ  disponibles dans l'Ã©tude principale.")
    args = parser.parse_args()
    root = ROOT / "output" / "backbone_source_factorial" / "source_only_tuning"
    commands: list[str] = []
    for source, dataset in SOURCES.items():
        for backbone_id, (backbone, prefix) in BACKBONES.items():
            for method in ("softtriple", "supcon", "cross_entropy"):
                if source == "btp" and backbone_id == "qwen3" and not args.include_qwen3_btp:
                    continue
                stem = f"{backbone_id}_{source}_{method}"
                base = yaml.safe_load((ROOT / "configs" / "methods" / ("supervised_macro_ft.yaml" if method == "cross_entropy" else f"{method}.yaml")).read_text(encoding="utf-8"))
                base["test_corpora"] = []
                base["source_corpus"] = source
                base.setdefault("data", {})["dataset_path" if method != "cross_entropy" else "data_csv"] = dataset
                base["data"]["dataset_path"] = dataset
                base.setdefault("model", {}).update({"backbone_name": backbone, "input_prefix": prefix})
                base_path = root / f"{stem}_base.yaml"; _write(base_path, base)
                grid_name = "supervised_macro_ft_grid.yaml" if method == "cross_entropy" else f"{method}_macro_ft_grid.yaml"
                grid = yaml.safe_load((ROOT / "configs" / "tuning" / grid_name).read_text(encoding="utf-8"))
                grid["base_config"] = str(base_path)
                grid["test_corpora"] = []
                grid["output_dir"] = f"output/backbone_source_factorial/source_only_tuning/{stem}"
                grid_path = root / f"{stem}_grid.yaml"; _write(grid_path, grid)
                relative_grid = grid_path.relative_to(ROOT).as_posix()
                command = (f"python scripts/tune_supervised_macro_ft.py --grid-config {relative_grid} --skip-final-fit"
                           if method == "cross_entropy" else
                           f"python scripts/tune_{method}_macro_ft.py --grid-config {relative_grid} --skip-final-fit")
                commands.append(command)
    (root / "commands.sh").write_text("#!/bin/bash\nset -euo pipefail\n" + "\n".join(commands) + "\n", encoding="utf-8")
    print(root / "commands.sh")


if __name__ == "__main__":
    main()
