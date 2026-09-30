import sys

import pandas as pd

sys.path.insert(0, "text/recurrent_scenarios")

from scenario_reporting import (  # noqa: E402
    build_bn_recurrence_display_table,
    select_bn_recurrence_display_scenarios,
)


def test_bn_recurrence_display_selection_is_directionally_balanced_and_traceable():
    scenarios = pd.DataFrame(
        {
            "scenario_id": [f"SC_{index:02d}" for index in range(10)],
            "upstream_labels": ["Work activity"] * 10,
            "B_label": ["Loss of balance"] * 10,
            "C_label": ["Fall"] * 10,
            "scenario_accident_count": [20, 18, 16, 14, 12, 10, 9, 8, 7, 6],
            "BN_expected_accident_count": [10.0, 11.0, 13.0, 14.1, 12.2, 13.0, 8.8, 8.1, 7.1, 5.9],
        }
    )

    selection = select_bn_recurrence_display_scenarios(
        scenarios,
        n_positive=2,
        n_negative=2,
        n_well_reproduced=2,
        n_dotplot=6,
    )
    scatter = selection["scatter"]
    dotplot = selection["dotplot"]
    table = build_bn_recurrence_display_table(selection)

    assert len(scatter) == 6
    assert (scatter.iloc[:2]["observed_minus_bn_count"] > 0).all()
    assert (scatter.iloc[2:4]["observed_minus_bn_count"] < 0).all()
    assert (scatter.iloc[4:]["scenario_accident_count"] >= scenarios["scenario_accident_count"].median()).all()
    assert len(dotplot) == 6
    assert set(scatter["display_id"]).issubset(set(table["ID"]))
    assert set(dotplot["display_id"]).issubset(set(table["ID"]))
    assert "Textual description" not in table.columns
    assert table["Scenario"].str.contains("\n", regex=False).all()
