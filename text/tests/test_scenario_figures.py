from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import pandas as pd

sys.path.insert(0, "text/recurrent_scenarios")

from scenario_figures import generate_scenario_figures


def test_generate_one_figure_per_scenario(tmp_path: Path):
    scenarios = pd.DataFrame(
        [
            {
                "scenario_id": "S1",
                "latent_family": "Z=1",
                "family_support": 0.4,
                "global_support": 0.2,
                "heading": "Context to event and consequence",
                "A0_label": "Work context",
                "A1_label": "Adverse condition",
                "B_label": "Event",
                "C_label": "Consequence",
                "factor_codes": "A0__T01;A1__T01;B__T01;C__T01",
            },
            {
                "scenario_id": "S2",
                "latent_family": "Z=2",
                "family_support": 25.0,
                "global_support": 10.0,
                "heading": "Context to consequence",
                "A0_label": "Work context",
                "A1_label": "",
                "B_label": "",
                "C_label": "Consequence",
                "factor_codes": "A0__T02;C__T02",
            },
        ]
    )
    normalized = generate_scenario_figures(scenarios, tmp_path)
    assert len(normalized) == 2
    assert len(list(tmp_path.glob("scenario_*.png"))) == 2
    assert len(list(tmp_path.glob("scenario_*.pdf"))) == 2
    assert (tmp_path / "scenarios_summary.csv").is_file()
