"""Generate result notebooks for every BTP macro-activity family."""

from __future__ import annotations

import json

from build_recurrent_scenarios_results_notebook import (
    BTP_FAMILY_DATASETS,
    NOTEBOOK_DIR,
    build_notebook as build_topic_notebook,
)
from build_scenario_mining_notebook import build_notebook as build_bn_notebook


def write_notebook(filename: str, notebook: dict) -> None:
    path = NOTEBOOK_DIR / filename
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Notebook written: {path}")


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_id in BTP_FAMILY_DATASETS:
        write_notebook(
            f"topic_modeling_results_{dataset_id}.ipynb",
            build_topic_notebook(dataset_id),
        )
        write_notebook(
            f"recurrent_scenarios_bn_analysis_{dataset_id}.ipynb",
            build_bn_notebook(dataset_id),
        )


if __name__ == "__main__":
    main()
