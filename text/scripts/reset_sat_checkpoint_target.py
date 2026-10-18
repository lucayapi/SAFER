#!/usr/bin/env python3
"""Clear exactly one checkpoint-selection target before a deliberate Slurm restart.

This helper lives outside sat_lodo_checkpoint_select.py so changing restart
behavior does not invalidate that trainer's already-passed smoke signature.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

RUN_NAME = "sat12l_sm_gold_v3_rq1_checkpoint_selection_v1"
CORPORA = ("btp", "metallurgie", "caou_chimie_plas", "bois_textille",
           "service_temporaire", "nicollin")


def reset_target(bundle: Path, run_root: Path, task_index: int) -> Path:
    if task_index not in range(len(CORPORA)):
        raise ValueError("Slurm task index must be in 0..5")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if tuple(manifest.get("fold_order", ())) != CORPORA:
        raise ValueError("Bundle corpus order differs from the Slurm array")

    # The run root is intentionally narrow. Never accept an arbitrary parent
    # passed by a caller, even if it happens to contain target_<corpus>.
    root = run_root.resolve(strict=True)
    if (not root.is_dir() or root.name != RUN_NAME or root.parent.name != "runs"
            or root.parent.parent.name != "safer_sat_lodo"):
        raise ValueError(f"Unexpected checkpoint-selection run root: {root}")
    target = CORPORA[task_index]
    run = root / f"target_{target}"
    if run.is_symlink() or getattr(run, "is_junction", lambda: False)():
        raise ValueError(f"Refusing to remove a linked target directory: {run}")
    if run.exists():
        if not run.is_dir() or os.path.ismount(run) or run.resolve(strict=True).parent != root:
            raise ValueError(f"Refusing to remove a non-target path: {run}")
        shutil.rmtree(run)
        print(f"Old target run removed: {run}", flush=True)
    else:
        print(f"Target run does not exist yet: {run}", flush=True)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    args = parser.parse_args()
    reset_target(args.bundle, args.run_root, args.task_index)


if __name__ == "__main__":
    main()
