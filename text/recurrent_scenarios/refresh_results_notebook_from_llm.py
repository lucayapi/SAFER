"""Refresh a results notebook from frozen factors and LLM labelling onward.

The script preserves executed cells before the frozen-factor reconstruction,
replaces the editable settings cell, and clears/replaces the reconstruction,
LLM and subsequent reporting cells from the current generated notebook
template. It never reruns theme discovery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK_DIR = ROOT / "notebooks"
FROZEN_SECTION_MARKER = "Frozen themes and audit dictionary"


def load_notebook(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def find_refresh_section(cells: list[dict]) -> int:
    for index, cell in enumerate(cells):
        text = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown" and FROZEN_SECTION_MARKER in text:
            return index
    raise ValueError(f"Frozen-factor section not found: {FROZEN_SECTION_MARKER}")


def first_code_cell(cells: list[dict]) -> int:
    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            return index
    raise ValueError("Notebook contains no code cell")


def refresh_notebook(dataset: str, *, restore_checkpoint: bool = False) -> Path:
    target = NOTEBOOK_DIR / f"topic_modeling_results_{dataset}.ipynb"
    checkpoint = NOTEBOOK_DIR / ".ipynb_checkpoints" / f"topic_modeling_results_{dataset}-checkpoint.ipynb"
    # Build the template in memory so the target may itself contain preserved
    # outputs from an older notebook layout.
    from build_recurrent_scenarios_results_notebook import build_notebook

    template = build_notebook(dataset)
    source = load_notebook(checkpoint if restore_checkpoint else target)

    template_start = find_refresh_section(template["cells"])
    source_start = find_refresh_section(source["cells"])
    source_cells = source["cells"][:source_start]
    # Older notebooks contained a manual PARTITION_SELECTIONS cell immediately
    # before the frozen-factor section. Its IDs and retired ``pareto/`` paths
    # conflict with the persisted selected_configurations.csv used below.
    source_cells = [
        cell
        for cell in source_cells
        if "PARTITION_SELECTIONS = {" not in "".join(cell.get("source", []))
    ]
    settings_index = first_code_cell(source_cells)
    template_settings_index = first_code_cell(template["cells"])
    source_cells[settings_index] = template["cells"][template_settings_index]

    source["cells"] = source_cells + template["cells"][template_start:]
    target.write_text(json.dumps(source, ensure_ascii=False, indent=1), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("caou", "btp", "metallurgie"))
    parser.add_argument(
        "--restore-checkpoint",
        action="store_true",
        help="Use the notebook's Jupyter checkpoint as the preserved pre-LLM portion.",
    )
    args = parser.parse_args()
    path = refresh_notebook(args.dataset, restore_checkpoint=args.restore_checkpoint)
    print(f"Notebook refreshed from LLM labelling onward: {path}")


if __name__ == "__main__":
    main()
