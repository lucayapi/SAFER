from theme_dictionary.select_feasible_config import choose_threshold


def test_selects_smallest_exactly_feasible_declared_threshold():
    config = {"model": {
        "max_theme_length": 4,
        "overlap_values": [3, 1],
        "max_groups_per_component": 1,
        "max_total_groups": 10,
    }}
    baskets = [(0, 1, 2, 3)]
    selected, attempts = choose_threshold(
        config, baskets, ["A0", "A1", "B", "C"], [1, 2]
    )
    assert selected == 2
    assert attempts[0]["overlap_checks"][0]["status"] == "resource_limit"
    assert [check["status"] for check in attempts[1]["overlap_checks"]] == [
        "feasible", "feasible"
    ]
